import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from generation import FiguresDataset, id_to_cname


def show_img(tensor_chw):
    tensor_hwc = tensor_chw.permute(1, 2, 0)
    plt.imshow(tensor_hwc)
    plt.show()


def pick(element):
    """Visualize an element from the dataset with all bounding boxes and figure labels"""
    tensor, description = element
    # read_image outputs uint8 0..255,
    # we transform that to float on the fly but still need uint8 for visualization to work
    tensor = torchvision.transforms.ConvertImageDtype(torch.uint8)(tensor)
    # [4, 194, 144, 28.7, 28.7]
    boxes_t, labels = [], []
    for f in description:
        # lay out bbox (xcen,ycen,w,h) as (xmin,ymin,xmax,ymax)
        bbox = (f[1] - f[3]/2, f[2] - f[4]/2, f[1] + f[3]/2, f[2] + f[4]/2)
        boxes_t.append(bbox)
        labels.append(id_to_cname[f[0]])
    tensor_w_boxes = torchvision.utils.draw_bounding_boxes(image=tensor, boxes=torch.tensor(boxes_t), labels=labels, colors='black')
    return tensor_w_boxes


def sample(elements, size=9):
    """Visualize a bunch of from the dataset with all bounding boxes and figure labels"""
    sample_list = [pick(elements[i]) for i in range(size)]
    show_img(torchvision.utils.make_grid(sample_list, nrow=np.sqrt(size).astype(int)))


if __name__ == '__main__':
    tr = torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
        # torchvision.transforms.ColorJitter(
        #         brightness=0.5, contrast=0.5,
        #         saturation=0.5, hue=0.5
        #     ),
        torchvision.transforms.Resize(416, antialias=None),
    ])
    ds = FiguresDataset(transforms=tr)
    # show_img(pick(ds[0]))
    sample(ds)
    print(ds[0][0])

##moved to jupyter notebook