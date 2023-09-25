"""Precision, Recall, PR-RC curve, AP, mAP for detecc"""

import torch

from aux_utils import raw_transform, iou_pairwise
from modules import ANCHORS
from generation import EPS

def ll_stats(pred_s, tar_s, scale, tp_iou_threshold):
    """called separately at each of 3 scales, torch-compliant,
    retrieves class_size, counts TP, FN, FP and masks successful predictions:
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
    # get iou for all preds with corresponding target bboxes
    ious_s = iou_pairwise(pred_s[..., 1:5], tar_s[..., 1:5])

    gpr = tar_s[..., 0] == 1  # GT presence mask (indices) = all positives (any class)
    ppr = pred_s[..., 0] == 1  # prediction presence mask (indices) = all predicted objects
    # put away target objects that haven't been predicted (0-score, any class, any box)
    wasted = gpr * ~ppr
    wasted_size = wasted.count_nonzero().item()
    # from now on, consider only the part of prediction that has detected some objects that could be target objects
    got = gpr * ppr
    # filter by IoU threshold, don't care about their class or object existence for now
    enough_iou = ious_s[..., 0] > tp_iou_threshold

    counts = []
    valid_predictions = False   # broadcasts to final shape
    for k in range(int(max(cl_indices))):
        # get precision and recall for each class as if there is nothing else
        ap_mask = tar_s[..., 5] == k  # all GT positives <--> GT class = K
        pp_mask = pred_s[..., 5] == k  # all positive predictions <--> predicted class = K

        TPm, FNm_ = got * ap_mask * pp_mask * enough_iou, got * (ap_mask * ~pp_mask + ap_mask * pp_mask * ~enough_iou)
        FPm_, TNm = got * ~ap_mask * pp_mask, got * ~ap_mask * ~pp_mask

        TP = pred_s[TPm].view(..., -1).shape[0]
        # sum up TP masks for all classes
        valid_predictions += TPm
        # add wasted targets to counts -- !check torch.stack etc. behaviour with empty arrays!
        FN = torch.stack(tuple(map(lambda t: t.view(..., -1), [pred_s[FNm_], tar_s[wasted * ap_mask]]))).shape[0]
        FP = torch.stack(tuple(map(lambda t: t.view(..., -1), [pred_s[FPm_], tar_s[wasted * ~ap_mask]]))).shape[0]

        class_size = (True + wasted) * ap_mask.count_nonzero().item()
        precision_k = TP / (TP + FP)
        recall_k = TP / (TP + FN)
        f1_score_k = (2 * precision_k * recall_k)/(precision_k + recall_k) if precision_k * recall_k != 0 else 0

        counts.append((TP, FN, FP, precision_k, recall_k, f1_score_k, class_size, wasted_size))
    return valid_predictions, ious_s, counts


def prc_plot(preds, tars, scales, iou_threshold):
    # we have [0..3] detections per object, i.e. it could be detected on some scale better, that means we should
    # find valid detections of same objects on different grids (different anchor, different grid size) and take best
    # the only way to do that is to extract identical target bbox from 3 tensors~(batch_size, # anchors, s, s, 6)

    # Instead I will treat these as (imaginary) 3 separate models each specializes at small/medium/large objects
    # my assumptions:
    # if small objects prevail, (ideal) S-model predicts all those but M and L do not have to predict any,
    # if equally distributed, all 3 models must predict something, each of 3 scores matters
    # thus their combined score should account for proportions of objects for each scale => weighed mean
    # ideally TP_L, TP_M, TP_S should approach #Large, #Medium, #Small => nice approximations for those

    # Let's wrap it with a tensor => shaped 3, 5, 8
    counts_ks = torch.tensor([ll_stats(preds[s], tars[s], scale=s, tp_iou_threshold=iou_threshold)[2] for s in scales])
    metrics_ks = counts_ks.permute(1, 0, 2)[..., 3:6]   # => shaped 5, 3, 8(3)
    scales_distribution = metrics_ks[..., 0]/metrics_ks[..., 0].sum(dim=1, keepdim=True)  # => shaped 5, 3
    metrics_kw = (metrics_ks * scales_distribution).sum(axis=1)   # 5, 3 broadcasts to 5, 3, 3 then sum 2nd dim => 5, 3

    # get ready for average precision for each class AP_k

















#TODO 2 get mAP