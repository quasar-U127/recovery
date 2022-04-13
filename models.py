from turtle import forward
from typing import List
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_small


class Simple(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        k = [n, n//8, 2]
        layers = []
        self.model = nn.Sequential()
        for i in range(len(k)-1):
            self.model.append(nn.Linear(k[i], k[i+1]))
            self.model.append(nn.ReLU())
        # self.layers = nn.Sequential(layers)
        # self.dense = nn.Linear(n,n//2)
        # self.dense1 = nn.Linear(n//2,2)

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(nn.Module):
    def __init__(self, sizes: List[int]) -> None:
        super().__init__()
        self.model = nn.Sequential()
        for i in range(len(sizes)-1):
            self.model.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                self.model.append(nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class MLPOneHot(nn.Module):
    def __init__(self, sizes: List[int], num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential()
        sizes[-1] = num_classes*sizes[-1]
        self.num_classes = num_classes
        for i in range(len(sizes)-1):
            self.model.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                self.model.append(nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        outputs = x.shape[-1]
        x = x.view(-1, self.num_classes, outputs//self.num_classes)
        return x


class ImageAlteration(nn.Module):
    def __init__(self, tail_sizes: List[int], num_classes: int) -> None:
        super().__init__()
        # self.mobilenet = mobilenet_v3_small()
        self.model = nn.Sequential()
        self.num_classes = num_classes
        self.model.append(mobilenet_v3_small())
        self.model.append(MLPOneHot(
            sizes=[1000]+tail_sizes, num_classes=num_classes))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
