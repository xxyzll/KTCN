import os
from .data import register_all_OWOD_dataset
from .rpn import FPNRPN
from .roi_heads import FPNROIHeads
from .rcnn import KTCN

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_OWOD_dataset(_root)

