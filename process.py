"""actual usage has been moved to jupyter notebook"""

import numpy as np
import torch
import torchvision
# template class for a dataset (makes our dataset compatible with torchvision)
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
from generation import load_dataset, id_to_cname, PATH


YOLO_SIZE = 416
# 3 feature maps at 3 different scales based on YOLOv3 paper
GRID_SIZES = (YOLO_SIZE // 32, YOLO_SIZE // 16, YOLO_SIZE // 8)
ANCHORS = (
    ((0.28, 0.22), (0.38, 0.48), (0.9, 0.78)),
    ((0.07, 0.15), (0.15, 0.11), (0.14, 0.29)),
    ((0.02, 0.03), (0.04, 0.07), (0.08, 0.06)),
)


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


class FiguresDataset(VisionDataset):
    def __init__(self, transforms, iou_threshold=0.5, root=PATH, anchors=ANCHORS, gs=GRID_SIZES):
        super().__init__(root)
        self.iou_thr = iou_threshold
        self.aug = transforms
        # anchors are set by just (relative) width & height, nested tuple 3*3
        self.anchors = anchors
        # number of anchors at each scale
        self.nan_per_scale = len(anchors)
        # each of 3 scales has grid_size and set of 3 anchors
        self.grid_sizes = gs
        assert self.nan_per_scale == len(self.grid_sizes), "#anchors doesn't coincide with #grid sizes"
        self.images, self.bboxes, self.c_idx = load_dataset(transforms=self.aug)
        assert len(self.images) == len(self.bboxes), 'wrong dataset generation, please retry'
        # establish 1:1 correspondence (at each scale): ground truth bounding box <-> anchor box and cell as target
        # let's keep track of bboxes that have never been mapped (on all 3 scales) for further investigation purposes
        self.unused_bboxes = []
        # self.targets = [self.build_targets(*p) for p in zip(self.bboxes, self.c_idx)]  # it's 1:1 within an image only!!
        # assert len(self.targets) == len(self.bboxes), 'wrong targets, check their builder method'
        # print(f'{len(self.targets)} targets at {len(self.grid_sizes)} scales have been created, '
        #       f'{len(self.unused_bboxes)} bounding boxes wasted')
        # self.mem_usage = (sys.getsizeof(self.images) // 2 ** 20, sys.getsizeof(self.targets) // 2 ** 20)
        # print(f"RAM usage images{self.mem_usage[0]} MB, targets{self.mem_usage[1]} MB")

    def __getitem__(self, i):
        try:
            transformed = self.aug(image=self.images[i], bboxes=self.bboxes[i], cidx=self.c_idx[i])
            targets = self.build_targets(transformed['bboxes'], transformed['cidx'])
            return transformed['image'], transformed['bboxes'], transformed['cidx'], targets
        except ValueError:
            print('error!', self.bboxes[i])
            return None

    def __len__(self):
        return len(self.images)

    def build_targets(self, bbox_list, label_list):
        """exclusively assigns 1 cell, 1 anchor (in that cell) to each bounding box at all 3 scales (if possible)"""
        targets = []
        for bb, ci in zip(bbox_list, label_list):
            found = None
            # extract current bounding box's (relative to image!) coordinates
            x, y, w, h = bb
            # prepare 3*s*s*(presence(0/1),bbox(4),class_id) torch tensor dummies for all scales
            target_dummies = [torch.zeros((self.nan_per_scale, s, s, 6)) for s in self.grid_sizes]
            for i, (al, gs, td) in enumerate(zip(self.anchors, self.grid_sizes, target_dummies)):
                found = False
                al_w_centers = [(x + a[0]/2, y + a[1]/2, *a)for a in al]
                # cell choice - put current s-grid onto original image, take a cell w/ bb center inside (if not taken)
                cx, cy = int(gs * x), int(gs * y)  # cell ~ top left corner relative (to grid) coordinates
                # anchor choice - 'best' (by IoU) of all free anchors at each step, suppress the rest w/ same role (NMS)
                anchor_ious = iou(base_box=bb, list_of_boxes=al_w_centers)
                # sort in descending manner and get sorting indices
                an_top = sorted(range(len(anchor_ious)), reverse=True, key=lambda _: anchor_ious[_])
                for ai in an_top:  # an_top = [2, 1, 0]
                    # check if the anchor at this cell is already used (=mapped to another bbox)
                    used = td[ai, cx, cy, 0]
                    if not used:
                        if not found:
                            # measure bbox in current grid's cells (new coordinates are relative to the cell)
                            width_c, height_c = gs * w, gs * h  # bbox covers that many cells like this one
                            shift_cx, shift_cy = gs * x - cx, gs * y - cy  # center is that shifted from tlc of the cell
                            td[ai, cx, cy, 0] = 1  # occupy
                            td[ai, cx, cy, 1:5] = torch.tensor([shift_cx, shift_cy, width_c, height_c])  # attach bb (1 scale)
                            td[ai, cx, cy, 5] = ci  # attach class id label, ensure its integer value
                            found = True
                        # NMS -- taken best, suppress rest (if free and detect this bbox quite good)
                        elif found and anchor_ious[ai] > self.iou_thr:
                            # best anchor is assigned, get rid of rest (only the ones that also detect this bbox ok!)
                            td[ai, cx, cy, 0] = -1  # ignore, i.e. they're NOT available for future (other bboxes)
                # replace target_dummy with td (even if td hasn't changed, i.e. all zeros)
                target_dummies[i] = td
            targets.append(target_dummies)
            # NB all anchors at cell might have been used up(by previous bboxes) => bbox is NOT detected on the scale gs
            if not found:  # found=None means that for-cycle has never started, that's impossible but nevertheless
                self.unused_bboxes.append(bb)
        return targets


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tr_list = [A.Normalize((0, 0, 0), (0.5, 0.5, 0.5)), A.Resize(416, 416), ToTensorV2()]
    tr = A.Compose(tr_list, bbox_params=A.BboxParams(format='yolo', label_fields=['cidx']))

    ds = FiguresDataset(transforms=tr)
    # show_img(pick(ds[0][:3]))
    sample(ds)
    print(ds[0][2])