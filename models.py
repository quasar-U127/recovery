from typing import List
from torch import Tensor, nn
import torch.nn.functional as F


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
