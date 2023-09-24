"""Precision, Recall, PR-RC curve, AP, mAP for detecc"""

import torch

from aux_utils import raw_transform, iou_pairwise
from modules import ANCHORS


def forward(pred_s, tar_s, scale):
    """called separately at each of 3 scales, torch-compliant"""
    # current anchor_s ~ 3*2 tensor, add dimensions to multiply freely
    anchors_s = torch.tensor(ANCHORS[scale]).reshape(1, 3, 1, 1, 2)

    # transform prediction to target bboxes format using given anchors
    tr_boxes = raw_transform(pred_s, anchors_s)[..., :5]  # 0-object score, 1-4 bbox sxsywh,
    # get single class index from class logits
    cl_indices = torch.argmax(pred_s[..., 5:], dim=-1)  # 5-9 class logits
    pred_s = torch.cat([tr_boxes, cl_indices], dim=-1)
    # compare with target bbox by iou
    ious = iou_pairwise(pred_s, tar_s)

    yobj = tar_s[..., 0] == 1  # GT presence mask (indices) = all positives (any class)
    nobj = tar_s[..., 0] == 0  # GT absence mask (indices) = all GT negatives (any class)

    # true positive: yobj
    # false positive:

    # consider only the part of prediction that may contain target objects
    pred_s = pred_s[yobj]
    tar_s = tar_s[yobj]



#TODO 1 split up classes, calculate PRC per each class
#TODO 2 get mAP