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
    # [[1, 16, 32, 76.10407640085654, 92.10407640085654], ...]
    boxes_t, labels = [], []
    for f in description:
        boxes_t.append(f[1:])
        labels.append(id_to_cname[f[0]])
    tensor_w_boxes = torchvision.utils.draw_bounding_boxes(image=tensor, boxes=torch.tensor(boxes_t), labels=labels, colors='black')
    return tensor_w_boxes


def sample(elements, size=9):
    """Visualize a bunch of from the dataset with all bounding boxes and figure labels"""
    sample_list = [pick(elements[i]) for i in range(size)]
    show_img(torchvision.utils.make_grid(sample_list, nrow=np.sqrt(size).astype(int)))


if __name__ == '__main__':
    ds = FiguresDataset()

    # show_img(pick(ds[0]))
    sample(ds)
    print(ds[0][1])

##moved jo jupyter notebook