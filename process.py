import numpy
import torch, torchvision

from generation import load_dataset, picture_path
images, data = load_dataset()

# make our dataset compatible with torchvision
class Figures(torchvision.datasets.VisionDataset):
    def __init__(self, root, json_path, transform=None):
        super().__init__(root)
        self.transform = transform





