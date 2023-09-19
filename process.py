"""actual usage has been moved to jupyter notebook"""
import torch
import torch.nn as nn
# template class for a dataset (makes our dataset compatible with torchvision)
from torchvision.datasets import VisionDataset

from generation import load_dataset, PATH, EPS
from aux_utils import iou, iou_pairwise, show_img, sample, pick

YOLO_SIZE = 416
# 3 feature maps at 3 different scales based on YOLOv3 paper
GRID_SIZES = (YOLO_SIZE // 32, YOLO_SIZE // 16, YOLO_SIZE // 8)
ANCHORS = (
    ((0.28, 0.22), (0.38, 0.48), (0.9, 0.78)),
    ((0.07, 0.15), (0.15, 0.11), (0.14, 0.29)),
    ((0.02, 0.03), (0.04, 0.07), (0.08, 0.06)),
)


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
                # cell choice - put current s-grid onto original image, take a cell w/ bb center inside (if not taken)
                cx, cy = int(gs * x), int(gs * y)  # cell ~ top left corner relative (to grid) coordinates
                al_w_centers = [(cx / gs + a[0] / 2, cy / gs + a[1] / 2, *a) for a in
                                al]  # anchor is centered within a cell
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
                            td[ai, cx, cy, 1:5] = torch.tensor(
                                [shift_cx, shift_cy, width_c, height_c])  # attach bb (1 scale)
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


def raw_transform(raw_net_out_s, anchors_s, partial=False):
    """this function is intended to be applied scalewise, partial=True limits this transformation to 1:3 ix;
    according to yolo design, this transforms (batched) raw NN output (just 4 coords) to the desired (target) format"""
    anchors = anchors_s.reshape(1, 3, 1, 1, 2)  # current anchor ~ 3*2 tensor, add dimensions to multiply freely
    raw_net_out_s[..., 1:3] = torch.sigmoid(raw_net_out_s[..., 1:3])
    if not partial:
        raw_net_out_s[..., 3:5] = torch.exp(raw_net_out_s[..., 3:5]) * anchors
    return raw_net_out_s


def targets_to_bboxes(tar_like: list, raw=False):
    """primary purpose of this function is to preprocess target/prediction for visualization,
    input looks like a list with 3=#GRID_SIZES tensors shaped (#anchors, gs(id), gs(id), 6), but
    those tensors are quite sparse, only their nonzero values correspond to bboxes and labels (of figures on image),
    but bboxes are yet to be decoded to absolute as they are given in relative (to grid, cell) format
        raw=True allows to process raw net outputs (i.e. predictions)
    NB0: it could account for 1st=batch dimension, but as far I am sure I won't apply this to batches
    NB1: we may have 0...3 bboxes instead of a single bbox from original dataset, because those bboxes might be
    already lost (as 'their' anchors+cells have been already taken) or we may have 1 bbox at all 3 scales
    NB2: [..., 0] = -1 is treated same way as zeros [..., 0] = 0 (for visualization at least)"""
    assert len(tar_like) == len(GRID_SIZES), f"This target doesn't have enough values for all {len(GRID_SIZES)} scales"
    boxes_dd = {}  # combine all information about boxes in a dict with keys = scale_id, values = (4-bbox, 1-label_id)
    for s in tar_like:
        # create presence mask (indices)
        present_s = tar_like[s][..., 0] == 1
        # convert 4 local (relative to cell shiftx, shifty, width, height) to absolute for all ai, cx, cy
        ix = torch.nonzero(present_s)[:, 1:3].float()  # 2D tensor #nonzero*(N-1) with values=indices, extract (cx, cy)
        # !this one should be cast to float manually because it's integer and going to be assigned to float!
        # copy tensor and replace 4 values of last dimension with absolute bbox coordinates there
        new_tl = tar_like[s].clone()  # .detach() maybe I should detach it too as it's for vis
        if raw:  # transform raw net outs to target first, unsqueeze-squeeze as it requires batch dimension
            tar_like[s] = raw_transform(tar_like[s].unsqueeze(0), ANCHORS[s]).squeeze(0)
        # calculate absolute center xy and assign at 1,2 positions of last dim (here 1:3 is just a coincidence)
        new_tl[present_s][..., 1:3] = (ix + tar_like[present_s][..., 1:3]) / GRID_SIZES[s]
        # calculate absolute width and height, then assign
        new_tl[present_s][..., 3:5] = tar_like[present_s][..., 3:5] / GRID_SIZES[s]
        # we don't need other dimensions anymore, reshape to (#found, 6) and save (params, labels) to dict
        boxes_dd[s] = (new_tl[present_s].reshape(-1, 6)[1:5], new_tl[present_s].reshape(-1, 6)[5])
        print(boxes_dd)
    return boxes_dd


class YOLOLoss(nn.Module):
    def __init__(self):
        """combined regressor/classifier loss"""
        super().__init__()
        self.mse = nn.MSELoss()  # for regression - box predictions
        self.bce = nn.BCEWithLogitsLoss()  # object presence/absence
        self.ent = nn.CrossEntropyLoss()  # for classes, we could use BCE but each box has just one class, no multilabel

        # losses are weighed (4 hyperparameters)
        self.la_cls = 1
        self.la_prs = 1
        self.la_abs = 10
        self.la_box = 10

    def forward(self, pred_s, tar_s, scale):
        """called separately at each of 3 scales, torch-compliant"""
        yobj = tar_s[..., 0] == 1  # presence mask (indices)
        nobj = tar_s[..., 0] == 0  # absence mask
        # NB: -1 value indices are completely ignored this way

        # has object loss: let object presence probability = iou_score, learns to predict not just 1 but own iou with gt
        anchors_s = ANCHORS[scale]
        # transform predictions to targets using given anchors
        pred_s = raw_transform(pred_s, anchors_s)
        # take the ones with object and compare with target bbox by iou
        ious = iou_pairwise(pred_s[..., 1:5][yobj], tar_s[..., 1:5][yobj]).detach()  # yet unsure about this detachment
        yo_loss = self.bce(pred_s[..., 0:1][yobj], ious * tar_s[..., 0:1][yobj])

        # bounding box loss: let's transform (part of) target to predictions, this trick allows better gradient flow,
        pred_s[..., 1:3] = raw_transform(pred_s, anchors_s, True)[..., 1:3]  # this part is same sigmoid as before
        tar_s[..., 3:5] = torch.log(EPS ** 2 + tar_s[..., 3:5]) / anchors_s  # transform target with inversion of before
        bo_loss = self.mse(pred_s[..., 1:5][yobj], tar_s[..., 1:5][yobj])

        # no object loss: 0:1 is a trick to keep dimensions and don't throw an error by mask
        no_loss = self.bce(pred_s[..., 0:1][nobj], tar_s[..., 0:1][nobj])
        # class loss: takes C logits, outputs single number, compared w/ target class
        ca_loss = self.ent(pred_s[..., 5:][yobj], tar_s[..., 5][yobj].long())

        return self.la_abs * no_loss + self.la_prs * yo_loss + self.la_box * bo_loss + self.la_cls * ca_loss


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tr_list = [A.Normalize((0, 0, 0), (0.5, 0.5, 0.5)), A.Resize(416, 416), ToTensorV2()]
    tr = A.Compose(tr_list, bbox_params=A.BboxParams(format='yolo', label_fields=['cidx']))

    # ds = FiguresDataset(transforms=tr)
    # # show_img(pick(ds[0][:3]))
    # sample(ds)
    # print(ds[0][2])

    # x = torch.tensor([[
    #     [ 0.6723,  1.2797],
    #      [ 1.5562, -0.3216],
    #      [-0.7927,  0.6416]],
    #
    #     [[ 0.3134,  0.2],
    #      [-0.8613, 0.5  ],
    #      [ 0.0538, -0.3507]],
    #
    #     [[-0.2787, -0.1952],
    #      [ 0.4247,  0.5],
    #      [ 1.2226,  0.5]]])
    #
    # y = torch.zeros_like(x)
    # mask = x[..., 1] == 0.5
    # print(torch.nonzero(mask))
    # print(torch.where(mask))
    # # y[mask] = torch.nonzero(mask).float()
    # # print(y)
