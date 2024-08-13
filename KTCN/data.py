import itertools
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager
import logging
import json
import torch

from detectron2.structures import BoxMode, Boxes, pairwise_iou
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm


logger = logging.getLogger('detectron2.KTCN.data')

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

COCO_CLASS_NAMES = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "couch", "train", "tv"
]

SOWOD_CATEGORIES = [
    # t1
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    # t2
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    # t3
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    # t4
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle",
    # Unknown
    "unknown"
]

OWOD_CATEGORIES = [
    # voc
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    # t2
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator", 
    # t3 
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", 
    # t4
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    # Unknown
    "unknown"
] 

def remove_previous(instances, prev_num):
    new_instance = []
    for instance in instances:
        if instance['category_id'] >= prev_num:
            new_instance.append(instance)
    return new_instance

def remove_unseen(instances, cur_seen):
    new_instance = []
    for instance in instances:
        if instance['category_id'] < cur_seen:
            new_instance.append(instance)
    return new_instance

def rename_unseen_to_unknown(instances, cur_seen):
    new_instance = []
    for instance in instances:
        if instance['category_id'] >= cur_seen:
            instance['category_id'] = len(OWOD_CATEGORIES)-1
        new_instance.append(instance)
    return new_instance

def load_one_sam_data(instances, sam_file_root, image_id, tr_sz=5, tr_iou=0.9):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_boxes = Boxes(torch.tensor([item['bbox'] for item in instances], device=device))
    with open(os.path.join(sam_file_root, f'{image_id}.json'), 'r') as f:
        sam_data = json.load(f)['box_result']
    sam_data = [item for item in sam_data if item['bbox'][2]>= tr_sz and item['bbox'][3]>= tr_sz]
    sam_boxes_list = [[
                        item['bbox'][0],item['bbox'][1], 
                        item['bbox'][0]+item['bbox'][2],
                        item['bbox'][1]+item['bbox'][3]]
                        for item in sam_data]  # xyxy format
    sam_score_list = [item['score'] for item in sam_data]
    sam_boxes = Boxes(torch.tensor(sam_boxes_list, device=device))
    ious, _ = pairwise_iou(sam_boxes, gt_boxes).max(dim=1)
    for boxe, iou, score in zip(sam_boxes_list, ious, sam_score_list):
        if(iou >= tr_iou):         
            continue
        instances.append({
            "category_id": len(OWOD_CATEGORIES)-1, 
            "bbox": [boxe[0], boxe[1], 
                    boxe[2], boxe[3]],  # xyxy
            "bbox_mode": BoxMode.XYXY_ABS,
            'score': score
        })
    return instances

def load_voc_coco_instances(dataset_root: str, file_name: str, class_names: Union[List[str], Tuple[str, ...]],
                            prev_num: int=0, cur_intro: int=20):
    cur_seen = prev_num + cur_intro
    
    dataset_root = os.path.join(dataset_root, "VOC2007")
    with PathManager.open(os.path.join(dataset_root, "ImageSets", "Main", file_name + ".txt")) as f:
        fileids = [file_name.strip() for file_name in f.readlines()]
        
    if class_names == OWOD_CATEGORIES:
        value_list = VOC_CLASS_NAMES
        index_list = COCO_CLASS_NAMES
    else:    
        value_list = COCO_CLASS_NAMES
        index_list = VOC_CLASS_NAMES
    annotation_dirname = os.path.join(dataset_root, "Annotations/")
    dicts = []
    logger.info('Loading OWOD dataset')
    for fileid in tqdm(fileids):
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dataset_root, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls_name = obj.find("name").text
            if cls_name in index_list:
                cls_name = value_list[index_list.index(cls_name)]
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            cls_id = class_names.index(cls_name)
      
            instances.append(
                {"category_id": cls_id, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        
        if 'train' in file_name:
            instances = remove_unseen(instances, cur_seen) 
            instances = remove_previous(instances, prev_num)
            instances = load_one_sam_data(instances, os.path.join(dataset_root, 'SAM'), fileid, tr_sz=25)
        if 'ft' in file_name:
            instances = remove_unseen(instances, cur_seen)
            instances = load_one_sam_data(instances, os.path.join(dataset_root, 'SAM'), fileid, tr_sz=25)
        if 'test' in file_name:
            instances = rename_unseen_to_unknown(instances, cur_seen)
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_all_OWOD_dataset(root):
    SPLITS = [
        ("t1_voc_coco_2007_train", "OWOD", "t1_train"),
        ("t2_voc_coco_2007_train", "OWOD", "t2_train"),
        ("t2_voc_coco_2007_ft", "OWOD", "t2_ft"),
        ("t3_voc_coco_2007_train", "OWOD", "t3_train"),
        ("t3_voc_coco_2007_ft", "OWOD", "t3_ft"),
        ("t4_voc_coco_2007_train", "OWOD", "t4_train"),
        ("t4_voc_coco_2007_ft", "OWOD", "t4_ft"),
        ("voc_coco_2007_test_1", "OWOD", "all_task_test"),
        ("voc_coco_2007_test_2", "OWOD", "all_task_test"),
        ("voc_coco_2007_test_3", "OWOD", "all_task_test"),
        ("voc_coco_2007_test_4", "OWOD", "all_task_test"),
    ]
    for split_name, split_type, file_name in SPLITS:
        register_OWOD_dataset(root, split_name, os.path.join(split_type, file_name))
        MetadataCatalog.get(split_name).evaluator_type = "owod"
    

def register_OWOD_dataset(root, split_name, file_name):
    class_names = OWOD_CATEGORIES

    if 't1' in split_name or 'test_1' in split_name:
        prev_num = 0
    elif 't2' in split_name or 'test_2' in split_name:
        prev_num = 20
    elif 't3' in split_name or 'test_3' in split_name:
        prev_num = 40
    elif 't4' in split_name or 'test_4' in split_name:
        prev_num = 60
 
    DatasetCatalog.register(
        split_name, lambda: load_voc_coco_instances(root, file_name, class_names, prev_num=prev_num))
    MetadataCatalog.get(split_name).set(
        thing_classes=list(class_names))
    MetadataCatalog.get(split_name).set(
        dataset_root=os.path.join(root, "VOC2007"))
    MetadataCatalog.get(split_name).set(
        file_name=file_name)
    

    
