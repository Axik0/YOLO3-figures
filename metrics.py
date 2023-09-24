"""Precision, Recall, PR-RC curve, AP, mAP for detecc"""

import torch

from aux_utils import raw_transform, iou_pairwise
from modules import ANCHORS


def prrc(pred_s, tar_s, scale, tp_iou_threshold):
    """called separately at each of 3 scales, torch-compliant
    take class K, split all target objects relative to that K vs rest(~K),
    filter out predicted objects by presence (PP), finally filter TP with IoU
    true positive: (K & PP & IoU, K)
    false negative: (~K & PP, K) | (~PP, K) | (K & PP & ~IoU, K)
    false positive: (K & PP, ~K)
    true negative:  (~K, ~K) | (~PP, ~K)"""
    # current anchor_s ~ 3*2 tensor, add dimensions to multiply freely
    anchors_s = torch.tensor(ANCHORS[scale]).reshape(1, 3, 1, 1, 2)

    gpr = tar_s[..., 0] == 1  # GT presence mask (indices) = all positives (any class)
    ppr = pred_s[..., 0] == 1  # prediction presence mask (indices) = all predicted objects

    # get target objects that haven't been predicted (0-score, any class, any box)
    wasted = tar_s[~ppr]
    # from now on, consider only the part of prediction that has detected objects and may contain target objects
    pred_s = pred_s[gpr * ppr]
    tar_s = tar_s[gpr * ppr]

    # transform prediction to target bboxes format using given anchors
    tr_boxes = raw_transform(pred_s, anchors_s)[..., :5]  # 0-object score, 1-4 bbox sxsywh,
    # get single class index from class logits
    cl_indices = torch.argmax(pred_s[..., 5:], dim=-1)  # 5 - class index
    pred_s = torch.cat([tr_boxes, cl_indices], dim=-1)

    prrc = []
    for k in range(int(max(cl_indices))):
        # get precision and recall for each class as if there is nothing else
        ap_mask = tar_s[..., 5] == k  # all GT positives <--> GT class = K
        pp_mask = pred_s[..., 5] == k  # all positive predictions <--> predicted class = K
        TP_, FN_ = pred_s[ap_mask * pp_mask], pred_s[ap_mask * ~pp_mask]    # have to be filtered by IOU and more
        FP_, TN = pred_s[~ap_mask * pp_mask], pred_s[~ap_mask * ~pp_mask]
        # compare with target bboxes by iou
        enough_iou = iou_pairwise(TP_, tar_s[ap_mask * pp_mask])[..., 0] > tp_iou_threshold
        TP = TP_[enough_iou].view(..., -1).shape[0]
        FN = torch.stack(tuple(map(lambda t: t.view(..., -1), [FN_, TP_[~enough_iou], wasted[ap_mask]]))).shape[0]
        FP = torch.stack(tuple(map(lambda t: t.view(..., -1), [FP_, wasted[~ap_mask]]))).shape[0]
        precision_k = TP / (TP + FP)
        recall_k = TP / (TP + FN)
        f1_score = (2 * precision_k * recall_k)/(precision_k + recall_k)
        prrc.append((precision_k, recall_k, f1_score))











#TODO 1 split up classes, calculate PRC per each class
#TODO 2 get mAP