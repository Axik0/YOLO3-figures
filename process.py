"""actual usage has been moved to jupyter notebook"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from generation import FiguresDataset, id_to_cname

YOLO_SIZE = 416


def show_img(tensor_chw):
    tensor_hwc = tensor_chw.permute(1, 2, 0)
    plt.imshow(tensor_hwc)
    plt.show()


def pick(element):
    """Visualize an element from the dataset with all bounding boxes and figure labels"""
    tensor, bboxes_, labels_ = element
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


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tr_list = [A.Normalize((0, 0, 0), (0.5, 0.5, 0.5)), A.Resize(416, 416), ToTensorV2()]
    tr = A.Compose(tr_list, bbox_params=A.BboxParams(format='yolo', label_fields=['cidx']))

    ds = FiguresDataset(transforms=tr)
    # show_img(pick(ds[0]))
    sample(ds)
    print(ds[0][2])

