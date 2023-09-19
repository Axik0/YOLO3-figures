"""various helper utilities for processing and visualization.
Eventually, all those should come in place"""

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
from generation import id_to_cname, EPS


def show_img(tensor_chw):
    tensor_hwc = tensor_chw.permute(1, 2, 0)
    plt.imshow(tensor_hwc)
    plt.show()


def pick(element):
    """Visualize an element from the dataset with all bounding boxes and figure labels"""
    tensor, (bboxes_, labels_, scale_idx) = element
    # we have floats but still need uint8 for this visualization to work
    tensor = torchvision.transforms.ConvertImageDtype(torch.uint8)(tensor)
    bboxes, labels = [], []
    for _ in range(len(labels_)):
        # lay out bbox (xcen,ycen,w,h) as (xmin,ymin,xmax,ymax)
        bb, label = bboxes_[_], labels_[_]
        bbox = list(map(lambda x: 416 * x, vertex_repr(bb)))
        bboxes.append(bbox)
        labels.append(id_to_cname[label])
    tensor_w_boxes = torchvision.utils.draw_bounding_boxes(image=tensor, boxes=torch.tensor(bboxes), labels=labels, colors=scale_idx)
    return tensor_w_boxes


def sample(elements, size=9):
    """Visualize a bunch of items from the dataset with all bounding boxes and figure labels"""
    sample_list = [pick(elements[i]) for i in range(size)]
    show_img(torchvision.utils.make_grid(sample_list, nrow=np.sqrt(size).astype(int)))


def vertex_repr(bbox):
    """converts from (x_center, y_center, w, h) to minimal-maximal vertex representation"""
    xc, yc, w, h = bbox
    vertex_min, vertex_max = (xc - w/2, yc - h/2), (xc + w/2, yc + h/2)
    return vertex_min, vertex_max


def iou(base_box, list_of_boxes):
    """treats bbox as (x_center, y_center, w, h), most likely normalized to [0..1],
    calculates areas of intersection and union for each box in the list, outputs list of IoU scores in [0..1]"""
    ious = []
    # represent boxes with their minmax vertices + calculate an area in a single pass
    vr = list(map(lambda bb: (*vertex_repr(bb), bb[2] * bb[3]), list_of_boxes + [base_box]))
    vmi_0, vma_0, area_0 = vr.pop()
    for b in vr:
        vmi_b, vma_b, area_b = b
        # get aoi, nearest max vertex - farthest min vertex
        dx = min(vma_0[0], vma_b[0]) - max(vmi_0[0], vmi_b[0])
        dy = min(vma_0[1], vma_b[1]) - max(vmi_0[1], vmi_b[1])
        aoi_0b = dx * dy if dx >= 0 and dy >= 0 else 0  # 0 means no intersection
        # get aou ~ sum of areas -- aoi (not to count twice)
        aou_0b = area_0 + area_b - aoi_0b
        assert aou_0b != 0, 'some of boxes have 0 width or height, incorrect input'
        iou_0b = aoi_0b/aou_0b
        ious.append(iou_0b)
    return ious


def iou_pairwise(tensor_1, tensor_2):
    """vectorized pairwise iou computation, returns tensor with all but last dimensions same, input
    tensors must have same shape & last dimension = 4 (describes a box)"""
    assert tensor_1.shape == tensor_2.shape, "wrong input tensors, shape mismatch"
    assert tensor_1.shape[-1] == 4, "last dimension is not 4, unable to process"
    # calculate areas
    area_1, area_2 = tensor_1[..., 2:3] * tensor_1[..., 3:4], tensor_2[..., 2:3] * tensor_2[..., 3:4]
    # switch to vertex_representation
    vmi_1, vma_1 = tensor_1[..., 0:2] - 0.5 * tensor_1[..., 2:4], tensor_1[..., 0:2] + 0.5 * tensor_1[..., 2:4]
    vmi_2, vma_2 = tensor_2[..., 0:2] - 0.5 * tensor_2[..., 2:4], tensor_2[..., 0:2] + 0.5 * tensor_2[..., 2:4]
    # get aoi, nearest max vertex - farthest min vertex, elementwise min max in torch
    dx = torch.minimum(vma_1[..., 0:1], vma_2[..., 0:1]) - torch.maximum(vmi_1[..., 0:1], vmi_2[..., 0:1])
    dy = torch.minimum(vma_1[..., 1:], vma_2[..., 1:]) - torch.maximum(vmi_1[..., 1:], vmi_2[..., 1:])
    aoi = dx * dy
    dx_n_dy_mask = dx + dy == 0
    aoi[dx_n_dy_mask] = 0  # mask out aoi when dx/dy are zero
    aou = area_1 + area_2 - aoi
    aou[area_1 + area_2 == 0] = EPS  # just in case, not to get zero-division
    return aoi/aou


def raw_transform(raw_net_out_s, anchors_s, partial=False):
    """this function is intended to be applied scalewise, partial=True limits this transformation to 1:3 ix;
    according to yolo design, this transforms (batched) raw NN output (just 4 coords) to the desired (target) format"""
    anchors = anchors_s.reshape(1, 3, 1, 1, 2)  # current anchor ~ 3*2 tensor, add dimensions to multiply freely
    raw_net_out_s[..., 1:3] = torch.sigmoid(raw_net_out_s[..., 1:3])
    if not partial:
        raw_net_out_s[..., 3:5] = torch.exp(raw_net_out_s[..., 3:5]) * anchors
    return raw_net_out_s


def tl_to_bboxes(tar_like: list, gs, anchors, raw=False):
    """primary purpose of this function is to preprocess target/prediction for visualization,
    input looks like a list with 3=#GRID_SIZES tensors shaped (#anchors, gs(id), gs(id), 6), but
    those tensors are quite sparse, only their nonzero values correspond to bboxes and labels (of figures on image),
    but bboxes are yet to be decoded to absolute as they are given in relative (to grid, cell) format
        raw=True allows to process raw net outputs (i.e. predictions)
    NB0: it could account for 1st=batch dimension, but as far I am sure I won't apply this to batches
    NB1: we may have 0...3 bboxes instead of a single bbox from original dataset, because those bboxes might be
    already lost (as 'their' anchors+cells have been already taken) or we may have 1 bbox at all 3 scales
    NB2: [..., 0] = -1 is treated same way as zeros [..., 0] = 0 (for visualization at least)"""
    assert len(tar_like) == len(gs), f"This target doesn't have enough values for all {len(gs)} scales"
    boxes, labels, scale_idx = [], [], []  # combine all information into 3 lists, bbox~(4-bbox, 1-label_id, 1-scale_id)
    for s in tar_like:
        # create presence mask (indices)
        present_s = tar_like[s][..., 0] == 1
        # convert 4 local (relative to cell shiftx, shifty, width, height) to absolute for all ai, cx, cy
        ix = torch.nonzero(present_s)[:, 1:3].float()  # 2D tensor #nonzero*(N-1) with values=indices, extract (cx, cy)
        # !this one should be cast to float manually because it's integer and going to be assigned to float!
        # copy tensor and replace 4 values of last dimension with absolute bbox coordinates there
        new_tl = tar_like[s].clone()  # .detach() maybe I should detach it too as it's for vis
        if raw:  # transform raw net outs to target first, unsqueeze-squeeze as it requires batch dimension
            tar_like[s] = raw_transform(tar_like[s].unsqueeze(0), anchors[s]).squeeze(0)
        # calculate absolute center xy and assign at 1,2 positions of last dim (here 1:3 is just a coincidence)
        new_tl[present_s][..., 1:3] = (ix + tar_like[present_s][..., 1:3]) / gs[s]
        # calculate absolute width and height, then assign
        new_tl[present_s][..., 3:5] = tar_like[present_s][..., 3:5] / gs[s]
        # we don't need other dimensions anymore, reshape to (#found, 6) and save (params, labels) to dict
        boxes_s, labels_s = new_tl[present_s].reshape(-1, 6)[1:5].tolist(), new_tl[present_s].reshape(-1, 6)[5].tolist()
        boxes.append(boxes_s)
        labels.append(labels_s)
        scale_idx += [s] * len(labels_s)    # denotes detection on different scales (by color or so)
    return boxes, labels, scale_idx


if __name__ == '__main__':
    box = (0.4, 0.4, 0.4, 0.2)  # xywh
    boxes = [(0.4, 0.4, 0.34, 0.19), (0.2, 0.4, 0.1, 0.3), (0.6, 0.5, 0.4, 0.2)]
    res = iou(box, boxes)
    print(res, sorted(range(len(res)), reverse=True, key=lambda _: res[_]))
    print(iou_pairwise(torch.rand(3, 3, 3, 4), torch.rand(3, 3, 3, 4)))
