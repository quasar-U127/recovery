from torch import nn
import torch.nn.functional as F

class Simple(nn.Module):
    def __init__(self,n:int) -> None:
        super().__init__()
        k = [n,n//2,2]
        self.layers = []
        
        self.dense = nn.Linear(n,n//2)
        self.dense1 = nn.Linear(n//2,2)

    def forward(self,x):
        x = self.dense(x)
        x = F.sigmoid(x)
        x = self.dense1(x)
        x = F.sigmoid(x)
        return x