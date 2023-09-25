"""Precision, Recall, PR-RC curve, AP, mAP for detecc"""

import torch

from aux_utils import raw_transform, iou_pairwise
from modules import ANCHORS


def pr_rc(pred_s, tar_s, scale, tp_iou_threshold):
    """called separately at each of 3 scales, torch-compliant,
    calculates class_size, precision, recall, f1_score (per class) metrics:
    taken class K, split all target objects relative to that K vs rest(~K),
    filter out predicted objects by presence (PP), finally filter TP with IoU
    true positive: (K & PP & IoU, K)
    false negative: (~K & PP, K) | (~PP, K) | (K & PP & ~IoU, K)
    false positive: (K & PP, ~K)
    true negative:  (~K, ~K) | (~PP, ~K)
    NB: boolean masks might be obviously excessive, that's intentional - for clarity

    """

    # current anchor_s ~ 3*2 tensor, add dimensions to multiply freely
    anchors_s = torch.tensor(ANCHORS[scale]).reshape(1, 3, 1, 1, 2)
    # transform prediction to target bboxes format using given anchors
    tr_boxes = raw_transform(pred_s, anchors_s)[..., :5]  # 0-object score, 1-4 bbox sxsywh,
    # get single class index from class logits
    cl_indices = torch.argmax(pred_s[..., 5:], dim=-1)  # 5 - class index
    pred_s = torch.cat([tr_boxes, cl_indices], dim=-1)

    gpr = tar_s[..., 0] == 1  # GT presence mask (indices) = all positives (any class)
    ppr = pred_s[..., 0] == 1  # prediction presence mask (indices) = all predicted objects
    # put away target objects that haven't been predicted (0-score, any class, any box)
    wasted = gpr * ~ppr
    # from now on, consider only the part of prediction that has detected some objects that could be target objects
    got = gpr * ppr

    prrc = []
    valid_predictions = False
    for k in range(int(max(cl_indices))):
        # get precision and recall for each class as if there is nothing else
        ap_mask = tar_s[..., 5] == k  # all GT positives <--> GT class = K
        pp_mask = pred_s[..., 5] == k  # all positive predictions <--> predicted class = K
        enough_iou = iou_pairwise(pred_s, tar_s)[..., 0] > tp_iou_threshold  # compare all preds with target bboxes
        TPm, FNm_ = got * ap_mask * pp_mask * enough_iou, got * (ap_mask * ~pp_mask + ap_mask * pp_mask * ~enough_iou)
        FPm_, TNm = got * ~ap_mask * pp_mask, got * ~ap_mask * ~pp_mask

        TP = pred_s[TPm].view(..., -1).shape[0]
        # add wasted targets to counts -- !check torch.stack etc. behaviour with empty arrays!
        FN = torch.stack(tuple(map(lambda t: t.view(..., -1), [pred_s[FNm_], tar_s[wasted * ap_mask]]))).shape[0]
        FP = torch.stack(tuple(map(lambda t: t.view(..., -1), [pred_s[FPm_], tar_s[wasted * ~ap_mask]]))).shape[0]

        class_size = (True + wasted) * ap_mask.count_nonzero().item()
        precision_k = TP / (TP + FP)
        recall_k = TP / (TP + FN)
        f1_score = (2 * precision_k * recall_k)/(precision_k + recall_k)
        prrc.append((class_size, precision_k, recall_k, f1_score))
        # sum up TP masks for all classes
        valid_predictions += TPm
    return prrc, valid_predictions


def prc_plot(preds, tars, scales, iou_threshold):
    for s in scales:
        prrc_s = pr_rc(preds[s], tars[s], tp_iou_threshold=iou_threshold)












#TODO 2 get mAP