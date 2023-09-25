"""Precision, Recall, PR-RC curve, AP, mAP for detecc"""

import torch

from aux_utils import raw_transform
from modules import ANCHORS


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

    gpr = tar_s[..., 0] == 1  # GT presence mask (indices) = all positives (any class)
    ppr = pred_s[..., 0] == 1  # prediction presence mask (indices) = all predicted objects
    # put away target objects that haven't been predicted (0-score, any class, any box)
    ignored = gpr * ~ppr
    ignored_size = ignored.count_nonzero().item()
    # from now on, consider only the part of prediction that has detected some objects that could be target objects
    got = gpr * ppr
    # filter by score threshold, don't care about their class or object existence for now
    enough_iou = pred_s[..., 0] >= tp_iou_threshold

    counts, prc = [], []
    valid_predictions = False   # broadcasts to final shape
    for k in range(int(max(cl_indices))):
        # get precision and recall for each class as if there is nothing else
        ap_mask = tar_s[..., 5] == k  # all GT positives <--> GT class = K
        pp_mask = pred_s[..., 5] == k  # all positive predictions <--> predicted class = K
        full_class_size = ap_mask.count_nonzero().item()

        # my TP and FN sum up to full class k size, the others treat pp_mask * ~enough_iou as FP for some ??reason??
        # my assumption: low obj score => class doesn't matter, model says no object (but should be)
        # their assumption: low obj score but ok class => model detected class, but we say "False" instead of model
        TPm, FNm = got * ap_mask * pp_mask * enough_iou, ap_mask * (got * (~pp_mask + pp_mask * ~enough_iou) + ignored)
        FPm, TNm = got * pp_mask * ~ap_mask, got * ~ap_mask * ~pp_mask + ignored * ~ap_mask

        # sum up TP masks for all classes
        valid_predictions += TPm

        # We want to plot PRC curve and its area is AP (for class k)
        # lay out as (#predictions, (score, is_TP, is_FP, is_FN)) - one of 3 booleans is always nonzero
        pred_state = torch.cat(tuple(map(lambda t: t[..., 0].view(-1, 1), [pred_s, TPm, FPm, FNm])), dim=-1)
        pred_state_nz = pred_state[pred_state.nonzero(as_tuple=True)]    # skip ALL zeros (NB not just zero score)
        p_sorting_indices = pred_state_nz[:, 0].argsort(dim=0, descending=True)    # sort by score
        acc_cm = torch.cumsum(pred_state_nz[p_sorting_indices][:, 1:], dim=0)  # capture dependence from score
        acc_precision = acc_cm[..., 1]/(acc_cm[..., 1] + acc_cm[..., 2])
        acc_recall = acc_cm[..., 1]/(acc_cm[..., 1] + acc_cm[..., 3])

        # get counts
        TPc = pred_s[TPm].view(-1, 6).shape[0]
        FPc = pred_s[TPm].view(-1, 6).shape[0]
        FNc = pred_s[FNm].view(-1, 6).shape[0]

        counts.append((TPc, FPc, FNc))
        prc.append((acc_precision, acc_recall))
    return counts, prc

import seaborn as sns

def prc_plot(preds, tars, scales):
    """state PR-curve, average precision for each class AP_k"""
    thresholds = torch.arange(0.5, 0.95, 0.05)     # AP@0.5:0.05:0.95
    prc_sk = torch.tensor([ll_stats(preds[s], tars[s], scale=s, tp_iou_threshold=0.5)[1] for s in scales])  # 3, 5, 2, N
    cts_sk = torch.tensor([ll_stats(preds[s], tars[s], scale=s, tp_iou_threshold=0.5)[0] for s in scales])
    s_dist = cts_sk[..., 0] / cts_sk[..., 0].sum(dim=0, keepdim=True)     # 3, 5 - tensor of scale distribution
    prc_k = (prc_sk * s_dist).sum(dim=0)   # 5, 2, N - tensor
    for k in range(prc_k.shape[0]):
        acc_pr, acc_rc = prc_k[k, 0, :], prc_k[k, 1, :]
        sns.lineplot(x=acc_rc, y=acc_pr)

def default_metrics(preds, tars, scales, iou_threshold, averaging='micro'):
    # we have [0..3] detections per object, i.e. it could be detected on some scale better, that means we should
    # find valid detections of same objects on different grids (different anchor, different grid size) and take best
    # the only way to do that is to extract identical target bbox from 3 tensors~(batch_size, # anchors, s, s, 6)

    # Instead I will treat these as (imaginary) 3 separate models each specializes at small/medium/large objects
    # my assumptions:
    # if small objects prevail, (ideal) S-model predicts all those but M and L do not have to predict any,
    # if equally distributed, all 3 models must predict something, each of 3 scores matters
    # thus their combined score should account for proportions of objects for each scale => weighed mean
    # ideally TP_L, TP_M, TP_S should approach #Large, #Medium, #Small => nice approximations for those

    # Let's wrap it with a tensor => shaped 3, 5, 3
    counts_sk = torch.tensor([ll_stats(preds[s], tars[s], scale=s, tp_iou_threshold=iou_threshold)[0] for s in scales])

    if averaging == 'micro':
        micro_c = counts_sk.sum(dim=1) #3, 3
        s_dist_micro = micro_c[:, 0]/micro_c[:, 0].sum() # broadcast to 3, 0
        avg_pr_micro = (micro_c[:, 0] / (micro_c[:, 0] + micro_c[:, 1]) * s_dist_micro).sum()
        avg_rc_micro = (micro_c[:, 0] / (micro_c[:, 0] + micro_c[:, 2]) * s_dist_micro).sum()
        avg_f1_micro = (2 * avg_pr_micro * avg_rc_micro) / (avg_pr_micro + avg_rc_micro) \
            if avg_pr_micro * avg_rc_micro != 0 else 0
        return avg_pr_micro, avg_rc_micro, avg_f1_micro
    elif averaging == 'macro':
        s_dist_macro = counts_sk[..., 0] / counts_sk[..., 0].sum(dim=0, keepdim=True)  # => shaped 3, 5
        avg_pr_macro = (counts_sk[..., 0] / (counts_sk[..., 0] + counts_sk[..., 1]) * s_dist_macro).sum(dim=0).mean()
        avg_rc_macro = (counts_sk[..., 0] / (counts_sk[..., 0] + counts_sk[..., 2]) * s_dist_macro).sum(dim=0).mean()
        avg_f1_macro = (2 * avg_pr_macro * avg_rc_macro) / (avg_pr_macro + avg_rc_macro) \
            if avg_pr_macro * avg_rc_macro != 0 else 0
        return avg_pr_macro, avg_rc_macro, avg_f1_macro
    else:
        print(f'check averaging method {averaging}')
        return counts_sk
