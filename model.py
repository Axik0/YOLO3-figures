import torch
import torch.nn as nn

from torchsummary import summary

IMAGE_SIZE = 416
NUM_CLASSES = 5
CFG = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53 - used as feature extractor
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, **kwargs):
        super().__init__()
        self.bn_state = batch_norm
        self.cv = nn.Conv2d(in_channels, out_channels, bias=not self.bn_state, **kwargs)
        # for predictions, we omit batch norm and relu activation
        self.bn = nn.BatchNorm2d(out_channels) if self.bn_state else nn.Identity()
        self.nl = nn.LeakyReLU(0.1) if self.bn_state else nn.Identity()

    def forward(self, x):
        return self.nl(self.bn(self.cv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, use=True, num_repeats=1):
        super().__init__()
        self.active = use
        self.repeated = num_repeats
        self.layers = nn.ModuleList()
        for r in range(self.repeated):
            self.layers += [nn.Sequential(
                CNNBlock(channels, channels // 2, kernel_size=1),
                CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
            )]

    def forward(self, x):
        for layer in self.layers:
            if self.active:
                return x + layer(x)
            else:
                return layer(x)


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.prediction = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 3 * (self.num_classes + 5), batch_norm=False, kernel_size=1),
        )

    def forward(self, x):
        # reshape n permute NCHW --> Nc1c2HW --> Nc1HWc2
        return (
            self.prediction(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2))


class YOLO3(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.cfg = CFG
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._auto_creation()

    def forward(self, x):
        outputs = []
        route_connections = []

        for lr in self.layers:
            if isinstance(lr, ScalePrediction):
                outputs.append(lr(x))
                continue
            x = lr(x)
            if isinstance(lr, ResidualBlock) and lr.repeated == 8:
                route_connections.append(x)
            if isinstance(lr, nn.Upsample):
                # concatenate channels
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _auto_creation(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in self.cfg:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == 'S':
                    layers.append(ResidualBlock(in_channels, use=False, num_repeats=1))
                    # we extract output and continue from this branch
                    layers.append(CNNBlock(in_channels, in_channels // 2, kernel_size=1))
                    layers.append(ScalePrediction(in_channels // 2, num_classes=self.num_classes))
                    in_channels = in_channels // 2
                if module == 'U':
                    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    in_channels = in_channels * 3
        return layers


if __name__ == "__main__":
    model = YOLO3(num_classes=NUM_CLASSES)
    test_x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    # print(model(test_x))
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
