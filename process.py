import numpy
import torch
import torchvision
from generation import load_dataset, json_path, picture_path


# make our dataset compatible with torchvision
class Figures(torchvision.datasets.VisionDataset):
    def __init__(self, root=picture_path, data_path=json_path, transform=None):
        super().__init__(root)
        self.paths, self.data = load_dataset(j_path=data_path, p_path=root, tv=True)
        self.images = [torchvision.io.read_image(path) for path in self.paths]
        self.transform = transform

    def __getitem__(self, idx):
        print(self.data[0])
        return self.images[idx]

    def __len__(self):
        return len(self.paths)


ds = Figures()
ds[0]