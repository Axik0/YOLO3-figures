import numpy
import torch
import torchvision
import matplotlib.pyplot as plt
from generation import FiguresDataset, id_to_cname


def show_img(tensor_chw):
    tensor_hwc = tensor_chw.permute(1, 2, 0)
    plt.imshow(tensor_hwc)
    plt.show()


def show(element):
    """Visualize an element from the dataset with all bounding boxes and figure labels"""
    tensor, description = element
    # (['Rhombus', 'Hexagon',...], [[16, 32, 76.10407640085654, 92.10407640085654], [20, 190, 71.8, 241.8],...])
    boxes_t, labels = torch.tensor(description[1]), [id_to_cname[i] for i in description[0]]
    tensor_w_boxes = torchvision.utils.draw_bounding_boxes(image=tensor, boxes=boxes_t, labels=labels, colors='black')
    show_img(tensor_w_boxes)


ds = FiguresDataset()
show(ds[0])

