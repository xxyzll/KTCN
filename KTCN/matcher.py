# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

from detectron2.layers import nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage

class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])])
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
            最大IOU匹配。不过添加了一个如果最大的IOU没有被选上那就添加到匹配集合中(allow_low_quality_matches)
            计算每一个框与GT的IOU，并取得最大的IOU作为它的匹配IOU。有4个阈值（0, 0.3, 0.7, 1）,根据匹配IOU的值计算label。计算方式为：
            0-0.3： 0（负样本）.
            0.3-0.7: -1(忽略).
            0.7~1：1(正样本）.
            匹配完成之后，如果某个GT最大的匹配框没有被选中为正样本，则使其为正样本。
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1



class simOTA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.steps = [8, 32, 64, 128, 256]
        self.box_transformer = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        
    def forward(
        self, **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """           
            return:
            matches List[Tensor(num_anchor)]: 每一个Anchor被匹配的gt_instances list索引
            match_labels List[Tensor(num_anchor)]: 每一个Anchor是否被匹配, 值为(-1, 0, 1)。-1忽略，0负样本，1正样本
        """
        # RPN 匹配
        if 'RPN_Matcher' in kwargs:
            return self.RPN_Matcher(**kwargs)
        if 'ROI_Matcher' in kwargs:
            return self.ROI_Mather(**kwargs)
            
    @torch.no_grad()
    def RPN_Matcher(self, anchors: List[Boxes], gt_instances: List[Instances], 
                    pred_anchor_deltas: List[torch.tensor], box2box_transform: Box2BoxTransform,
                    steps: List, cls_preds: List[torch.tensor], ignore: bool=True,
                    **kwargs):
        """
            anchors List[Box(num_anchor, 4)]:  
                每一个List的item为一个特征等级。
            gt_instances List[Instances{gt_classes: tensor, gt_boxes: tensor}]: 
                每一个List的item为一张图片的样本数量。
            
            return:
            matches List[Tensor(num_anchor)]: 每一个Anchor被匹配的gt_instances list索引
            match_labels List[Tensor(num_anchor)]: 每一个Anchor是否被匹配, 值为(-1, 0, 1)。-1忽略，0负样本，1正样本
        """
        x_shifts = torch.cat([(anchor.tensor[:, 0]+anchor.tensor[:, 2])/2 for anchor in anchors], dim=-1)
        y_shifts = torch.cat([(anchor.tensor[:, 1]+anchor.tensor[:, 3])/2 for anchor in anchors], dim=-1)
        expanded_strides = torch.cat([torch.full((anchors[feature_level].tensor.shape[0],), steps[feature_level]) \
                                                for feature_level in range(len(anchors))])
        anchors = Boxes.cat(anchors).tensor
        
        matched_bboxes, matched_labels = [], []

        for img_id in range(len(gt_instances)):
            per_img_deltas = torch.cat([pred_anchor_delta[img_id] 
                                        for pred_anchor_delta in pred_anchor_deltas], dim=0)
            per_img_gt_boxes, per_img_gt_classes = gt_instances[img_id].gt_boxes.tensor, \
                                                    gt_instances[img_id].gt_classes
            per_img_pred_boxes = box2box_transform.apply_deltas(per_img_deltas, anchors)
            num_gt, total_num_anchors = len(per_img_gt_boxes), len(per_img_pred_boxes)
            per_img_cls_preds = torch.cat([cls_pred[img_id] 
                                        for cls_pred in cls_preds], dim=0)
            
            (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg, ignore_mask) = \
            self.RPN_get_assignment(batch_idx=img_id, num_gt=num_gt, total_num_anchors=total_num_anchors,
                gt_bboxes_per_image=per_img_gt_boxes, gt_classes=per_img_gt_classes,
                bboxes_preds_per_image=per_img_pred_boxes, expanded_strides=expanded_strides,
                x_shifts=x_shifts, y_shifts=y_shifts, cls_preds=per_img_cls_preds,
                obj_preds=None, anchors=anchors
            )
            
            matched_label = torch.zeros_like(x_shifts)
            matched_label[fg_mask] = 1
            if ignore:
                bg_mask = matched_label[~fg_mask]
                bg_mask[ignore_mask] = -1
                matched_label[~fg_mask] = bg_mask
            
            matched_gt_boxes_i = torch.zeros_like(anchors)
            matched_gt_boxes_i[fg_mask] = per_img_gt_boxes[matched_gt_inds]
            matched_labels.append(matched_label)
            matched_bboxes.append(matched_gt_boxes_i)
            
            storage = get_event_storage()
            storage.put_scalar("matched IOU", sum(pred_ious_this_matching)/num_fg)
        return matched_labels, matched_bboxes
        
    def RPN_get_assignment(
        self,
        batch_idx: int,                 # 索引
        num_gt: int,                    # 当前图像有多少个gt box
        total_num_anchors: int,         # 有多少Anchor
        gt_bboxes_per_image: Boxes,     # 当前图像的gt_box
        gt_classes: torch.tensor,       # gt_box对应的cls label
        bboxes_preds_per_image: Boxes,  # 当前的预测box
        expanded_strides: torch.tensor, # 每个像素的缩放或预测之间的步长
        x_shifts: torch.tensor,         # 每个特征点在原特征图上的x坐标
        y_shifts: torch.tensor,         # 每个特征点在原特征图上的y坐标
        cls_preds: torch.tensor,        # 每个框的预测类别(B，8400，num_class)
        obj_preds: torch.tensor,        # 每个预测框的置信度。(B，8400，1)
        anchors: torch.tensor,
        set_iou_ignore=(0.3, 0.7),      # 设置IOU大于0.3小于0.7的为忽略
    ):
        """
            gt_matched_classes tensor(num_match): 
                过滤后匹配的类别索引, 如46
            fg_mask tensor(num_pred): 
                过滤的mask
            pred_ious_this_matching tensor(num_match): 
                当前匹配这些框所对应的IOU
            matched_gt_inds tensor(num_match): 
                过滤后匹配的的gt_instances索引, 如(0, 1 ... )
            num_fg int:
                num_match有多少正样本
        """
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        all_ious = pairwise_iou(Boxes(gt_bboxes_per_image), Boxes(anchors))
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        # 计算Box损失
        pair_wise_ious = pairwise_iou(Boxes(gt_bboxes_per_image), Boxes(bboxes_preds_per_image))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        # 计算CLS损失
        gt_cls_per_image = torch.ones_like(cls_preds_)
        cls_preds_ = (
            cls_preds_.float().sigmoid_()
        )        # 需要？
        pair_wise_cls_loss = F.binary_cross_entropy(
            input=cls_preds_,
            target=gt_cls_per_image,
            reduction="none"
        )
        del cls_preds_
        cost = (
            pair_wise_cls_loss[None, :]
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        ignore_mask = all_ious[:, fg_mask == 0].max(dim=0)[0] >= set_iou_ignore[0]
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss, all_ious
        
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
            ignore_mask,
        )
    
    def ROI_Mather(self, proposals: List[Instances], targets: List[Instances], **kwargs):
        """
            proposals: 
                每一个图片候选的list. 每个Instances中含有:
                    proposal_boxes: Boxes(1000, 4). 
                    objectness_logits: tensor(1000)
            predictions:
                这些proposals对应的预测

            targets:
                每一个图片的target. 每个target中含有:
                    gt_boxes: Boxes(num_gt, 4). 
                    gt_classes: tensor(num_gt)
            return:
                List[Instances]:
                每一个图片的候选匹配. 候选中含有:
                    proposal_boxes: Boxes(sampled_num, 4) 采样匹配后对应的候选boxes
                    objectness_logits: tensor(sampled_num) 采样匹配后对应的候选objectness
                    gt_classes: tensor(sampled) 采样匹配后对应的gt_classes, 其中self.num_class表示负样本
                    gt_boxes: Boxes(sampled_num, 4)  采样匹配后对应的gt_boxes, 仅当gt_classes != self.num_class有效
        """
        ret_proposals = []
        for per_img_proposals, per_img_targets in zip(proposals, targets):
            
            """分离出未知类别. with_unknown: 是否使用未知类别. same_matcher: 是否同时匹配 """
            if 'with_unknown' in kwargs and 'same_matcher' not in kwargs:
                unknown_targets = per_img_targets[per_img_targets.gt_classes == 80]
                known_targets = per_img_targets[per_img_targets.gt_classes != 80]
            else:
                known_targets = per_img_targets
            
            fg_mask, matched_gt_inds, pred_ious_this_matching, num_fg = \
                self.simota_matching_(per_img_proposals, known_targets)

                
            gt_classes = torch.full(
                per_img_proposals.objectness_logits.shape,
                81,
                device=per_img_proposals.objectness_logits.device
            )
            gt_classes[fg_mask] = known_targets.gt_classes[matched_gt_inds]
            gt_boxes = per_img_proposals.proposal_boxes.tensor
            gt_boxes[fg_mask] = per_img_targets.gt_boxes.tensor[matched_gt_inds]
            
            if 'with_unknown' in kwargs and 'same_matcher' not in kwargs and len(unknown_targets) > 0:
                bg_proposal = per_img_proposals[~fg_mask]
                bg_matched_mask, bg_matched_gt_inds, _, _ = \
                    self.simota_matching_(bg_proposal, unknown_targets)
                bg_matched_mask = bg_matched_mask.nonzero().squeeze(1)
                gt_classes[~fg_mask] = gt_classes[~fg_mask].scatter_(0, 
                                                                  bg_matched_mask, 
                                                                  unknown_targets.gt_classes[bg_matched_gt_inds])
                bg_boxes = gt_boxes[~fg_mask]
                bg_boxes[bg_matched_mask] = unknown_targets.gt_boxes.tensor[bg_matched_gt_inds]
                gt_boxes[~fg_mask] = bg_boxes
            per_img_proposals.gt_classes = gt_classes
            per_img_proposals.gt_boxes = Boxes(gt_boxes)
            ret_proposals.append(per_img_proposals)
        
        return ret_proposals
       
    def simota_matching_(self, proposal: Instances, targets: Instances, **kwargs):
        """
            proposal: 
                proposal_boxes: Boxes(num_proposal, 4). 
                objectness_logits: tensor(num_proposal)
            targets:
                gt_boxes: Boxes(num_gt, 4). 
                gt_classes: tensor(num_gt)
            predictions:
                cls_preds: tensor(num_proposal, num_classes)
                delta_preds: tensor(num_proposal*num_classes, 4)
            return:
                fg_mask: tensor(num_proposal)
                matched_gt_inds: tensor(num_matched)
                pred_ious_this_matching: tensor(num_matched)
                num_fg: int
        """
        """bbox loss"""
        num_proposal, num_gt, device = len(proposal), len(targets), proposal.objectness_logits.device
        # (num_known_gt, num_proposal)
        pair_wise_ious = pairwise_iou(targets.gt_boxes, proposal.proposal_boxes)                            
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        """objectness loss"""
        object_ness = proposal.objectness_logits.sigmoid()
        object_loss = F.binary_cross_entropy(object_ness, torch.ones_like(object_ness), reduction="none")
        cost = (
                object_loss[None, :]
                + 3.0 * pair_wise_ious_loss
        )
        fg_mask = torch.ones(num_proposal, dtype=torch.bool, device=device)
        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds) = \
            self.simota_matching(cost, pair_wise_ious, targets.gt_classes, num_gt, fg_mask)
            
        return [fg_mask, matched_gt_inds, pred_ious_this_matching, num_fg]
            
    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    
    def get_geometry_constraint(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts):
        """
            gt_bboxes_per_image Tensor(num_gt, 4): xyxy format
            expanded_strides Tensor(num_pred_boxes): 每一个box对应特征图的下采样率
            x_shifts Tensor(num_pred_boxes): 每一个anchor box的中心x
            y_shifts Tensor(num_pred_boxes): 每一个anchor box的中心y,
            
            return:
                anchor_filter tensor: 每一个Anchor是否在某个GT的中心center_dist*2的正方形内
                
        """
        """计算x, y中心"""
        x_centers_per_image = (x_shifts + 0.5)
        y_centers_per_image = (y_shifts + 0.5)
        gt_bbox_center_x = ((gt_bboxes_per_image[:, 0]+gt_bboxes_per_image[:, 2])/2).unsqueeze(1)
        gt_bbox_center_y = ((gt_bboxes_per_image[:, 1]+gt_bboxes_per_image[:, 3])/2).unsqueeze(1)
        
        # 在每个gt的中心裁剪出一个center_dist*2的正方形，判断anchor是否在这个里面
        center_radius=1.5
        center_dist = (expanded_strides * center_radius).to(gt_bbox_center_x.device)
        gt_bboxes_per_image_l = gt_bbox_center_x - center_dist
        gt_bboxes_per_image_r = gt_bbox_center_x + center_dist
        gt_bboxes_per_image_t = gt_bbox_center_y - center_dist
        gt_bboxes_per_image_b = gt_bbox_center_y + center_dist
        
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation