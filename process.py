import numpy
import torch
import torchvision
from generation import FiguresDataset

ds = FiguresDataset()
print(ds[0])