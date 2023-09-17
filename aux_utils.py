"""various helper utilities for processing and visualization.
Eventually, all those should come in place"""

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
from generation import id_to_cname, PATH, EPS
from process import YOLO_SIZE


def show_img(tensor_chw):
    tensor_hwc = tensor_chw.permute(1, 2, 0)
    plt.imshow(tensor_hwc)
    plt.show()


def pick(element):
    """Visualize an element from the dataset with all bounding boxes and figure labels"""
    tensor, bboxes_, labels_ = element[:3] # to comply with target at 4-th position
    # read_image outputs uint8 0..255,
    # we transform that to float on the fly but still need uint8 for visualization to work
    tensor = torchvision.transforms.ConvertImageDtype(torch.uint8)(tensor)
    bboxes, labels = [], []
    for _ in range(len(labels_)):
        # lay out bbox (xcen,ycen,w,h) as (xmin,ymin,xmax,ymax)
        bb, label = bboxes_[_], labels_[_]
        bbox = list(map(lambda x: YOLO_SIZE * x, (bb[0] - bb[2]/2, bb[1] - bb[3]/2, bb[0] + bb[2]/2, bb[1] + bb[3]/2)))
        bboxes.append(bbox)
        labels.append(id_to_cname[label])
    tensor_w_boxes = torchvision.utils.draw_bounding_boxes(image=tensor, boxes=torch.tensor(bboxes), labels=labels, colors='black')
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