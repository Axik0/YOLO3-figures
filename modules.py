"""actual usage has been moved to jupyter notebook"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from generation import load_dataset, PATH, EPS
from aux_utils import iou_pairwise, raw_transform, show_img, sample, pick, get_anchors

YOLO_SIZE = 416
# 3 feature maps at 3 different scales based on YOLOv3 paper
GRID_SIZES = (YOLO_SIZE // 32, YOLO_SIZE // 16, YOLO_SIZE // 8)
# ANCHORS = (
#     ((0.28, 0.22), (0.38, 0.48), (0.9, 0.78)),
#     ((0.07, 0.15), (0.15, 0.11), (0.14, 0.29)),
#     ((0.02, 0.03), (0.04, 0.07), (0.08, 0.06)),
# )

ANCHORS = (
    ((0.22838012478569927, 0.37234857956035894),
     (0.30127089816065944, 0.30470957552855793),
     (0.36307149222721874, 0.36530570814002944)),

    ((0.19109910688669632, 0.2804814482464234),
     (0.2570033528878385, 0.25741008305321017),
     (0.3500616294989826, 0.20384630961828798)),

    ((0.13782940739183006, 0.13803015194462828),
     (0.17738931739948338, 0.17951509178104036),
     (0.21803462084789216, 0.21450175377012273))
)

DEFAULT_TR = [A.Resize(YOLO_SIZE, YOLO_SIZE), ToTensorV2()]

LOSS_PARTS = ('absence', 'presence', 'bbox', 'class')

# no augmentations, just resized 256 --> 416, cast to torch.float tensors...(NB! order matters)


class FiguresDataset(Dataset):
    """Main task is to transform images and create yolo targets based on data loaded from PATH,
    must-have transforms are built-in already thus aug_list is supposed to contain some augmentations only
    part=(start_id, end_id) translates to slicing on loaded data
    upd_stats allows to get actual mean and standard deviation values per img channel, very slow"""

    def __init__(self, aug=(), part=slice(None, None), iou_threshold=0.5, anchors=ANCHORS, gs=GRID_SIZES,
                 upd_stats=False):
        super().__init__()
        self.iou_thr = iou_threshold
        # anchors are set by just (relative) width & height, nested tuple 3*3
        self.anchors = anchors
        # number of anchors at each scale
        self.nan_per_scale = len(anchors)
        # each of 3 scales has grid_size and set of 3 anchors
        self.grid_sizes = gs
        assert self.nan_per_scale == len(self.grid_sizes), "#anchors doesn't coincide with #grid sizes"
        self.images, self.bboxes, self.c_idx, mean_c, std_c = load_dataset(transforms=None, part_slice=part,
                                                                           stats=upd_stats)
        assert len(self.images) == len(self.bboxes), 'wrong dataset generation, please retry'
        # calculate per-channel mean and std (over whole dataset, slow) while loading or just use pre-calculated
        self.mean, self.std = (mean_c, std_c) if upd_stats else [0.641, 0.612, 0.596], [0.115, 0.11, 0.11]
        # aug_list should contain just image augmentations, the rest is already given
        self.tra = A.Compose(transforms=tuple(aug) + (A.Normalize(self.mean, self.std), *DEFAULT_TR),
                             bbox_params=A.BboxParams(format='yolo', label_fields=['cidx'], min_visibility=0.5))
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
            transformed = self.tra(image=self.images[i], bboxes=self.bboxes[i], cidx=self.c_idx[i])
            targets = self.build_targets(transformed['bboxes'], transformed['cidx'])
            return transformed['image'], targets
        except ValueError:
            print('error!', self.bboxes[i])
            return None

    def __len__(self):
        return len(self.images)

    def build_targets(self, bbox_list, label_list):
        """exclusively assigns 1 cell, 1 anchor (in that cell) to each bounding box at all 3 scales (if possible)"""
        target_dummies = [torch.zeros((self.nan_per_scale, s, s, 6)) for s in self.grid_sizes]
        for bb, ci in zip(bbox_list, label_list):
            found = None
            # extract current bounding box's (absolute!) coordinates
            x, y, w, h = bb
            # prepare 3*s*s*(presence(0/1),bbox(4),class_id) torch tensor dummies for all scales
            for i, (al, gs, td) in enumerate(zip(self.anchors, self.grid_sizes, target_dummies)):
                found = False
                # cell choice - put current s-grid onto original image, take a cell w/ bb center inside (if not taken)
                cx, cy = int(gs * x), int(gs * y)  # cell ~ top left corner relative (to grid,i.e. in cells) coordinates
                al_w_centers = [(cx / gs + a[0] / 2, cy / gs + a[1] / 2, *a) for a in
                                al]  # anchor is centered within a cell, this line adds center's absolute coordinates
                # anchor choice - 'best' (by IoU) of all free anchors at each step, suppress the rest w/ same role (NMS)
                anchor_ious = (iou_pairwise(torch.tensor(bb).unsqueeze(0), torch.tensor(al_w_centers))
                               .squeeze(0, -1).tolist())
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
                        target_dummies[i][ai, cx, cy] = td[ai, cx, cy]
            # NB all anchors at cell might have been used up(by previous bboxes) => bbox is NOT detected on the scale gs
            if not found:  # found=None means that for-cycle has never started, that's impossible but nevertheless
                self.unused_bboxes.append(bb)
        return tuple(target_dummies)


class YOLOLoss(nn.Module):
    def __init__(self, l_abs=10, l_prs=10, l_box=1, l_cls=1):
        """weighed regressor & classifier loss, consists of 4 parts"""
        super().__init__()
        self.mse = nn.MSELoss()  # for regression - box predictions
        self.bce = nn.BCEWithLogitsLoss()  # object presence/absence
        self.ent = nn.CrossEntropyLoss()  # for classes, we could use BCE but each box has just one class, no multilabel

        # losses are weighed (4 hyperparameters)
        self.la_abs = l_abs  # 10 originally
        self.la_prs = l_prs
        self.la_box = l_box  # 10 originally
        self.la_cls = l_cls

        self.id_to_parts = LOSS_PARTS

        self.combo_loss = torch.tensor(0.)

    def forward(self, pred_s, tar_s, scale):
        """called separately at each of 3 scales, torch-compliant"""
        yobj = tar_s[..., 0] == 1  # presence mask (indices)
        nobj = tar_s[..., 0] == 0  # absence mask
        # NB: -1 value indices are completely ignored this way

        # current anchor_s ~ 3*2 tensor, add dimensions to multiply freely
        anchors_s = torch.tensor(ANCHORS[scale]).reshape(1, 3, 1, 1, 2)
        # transform predictions to targets using given anchors, clone to avoid changes of an original tensor
        pred_st = raw_transform(pred_s, anchors_s)

        # has object loss: let object presence probability = iou_score, learns to predict not just 1 but own iou with gt
        # take the ones with object and compare with target bbox by iou
        ious = iou_pairwise(pred_st[..., 1:5][yobj], tar_s[..., 1:5][yobj])#.detach()  # yet unsure about this detachment
        yo_loss = self.bce(pred_st[..., 0:1][yobj], ious * tar_s[..., 0:1][yobj])

        # bounding box loss: let's transform (part of) target to predictions, this trick allows better gradient flow,
        pred_st_part = torch.cat([pred_st[..., 1:3], pred_s[..., 3:5]], dim=-1)  # part is same sigmoid as before
        tar_st_part = torch.cat([tar_s[..., 1:3], torch.log(EPS ** 3 + tar_s[..., 3:5]) / anchors_s],
                                dim=-1)  # inverse transform part of target
        bo_loss = self.mse(pred_st_part[yobj], tar_st_part[yobj])

        # no object loss: 0:1 is a trick to keep dimensions and don't throw an error by mask
        no_loss = self.bce(pred_s[..., 0:1][nobj], tar_s[..., 0:1][nobj])
        # class loss: takes C logits, outputs single number, compared w/ target class
        ca_loss = self.ent(pred_s[..., 5:][yobj], tar_s[..., 5][yobj].long())

        # stack 4 separate current loss values into 4-tensor
        self.combo_loss = torch.stack((self.la_abs * no_loss,
                                       self.la_prs * yo_loss,
                                       self.la_box * bo_loss,
                                       self.la_cls * ca_loss))

        return torch.sum(self.combo_loss)

    def get_state(self):
        return self.combo_loss.detach()


if __name__ == '__main__':
    ds = FiguresDataset(part=slice(*(200, 300)), upd_stats=False)
    # it's weird to move yolo constants to aux_utils, can't be imported from here as it leads to circular import error
    gsa = {'gs': GRID_SIZES, 'anchors': ANCHORS}
    show_img(pick(ds[0], **gsa))
    sample(ds, **gsa)
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
    # y[mask[:,1]] = 5

    # print(y[mask])
    # loss_f = YOLOLoss()
    # test = ds[0][1][1].unsqueeze(0)
    # test2 = ds[1][1][1].unsqueeze(0)
    # print(loss_f(test, test, 1))
    # we have to disable CE part as we don't have class probabilities
    # At least bbox part is zero
