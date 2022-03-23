from glob import escape, glob
import os
from torch.utils.data import Dataset
import numpy as np
from typing import Any, Tuple

class UndoData(Dataset):
    def __init__(self, path:str):
        self.root = path
        self.samples = [os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(escape(self.root),"*.npz"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: Any) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        item =  self.samples[index]
        sample = np.load(os.path.join(self.root,f"{item}.npz"))
        return sample["arr_0"], sample["arr_1"], sample["arr_2"]
