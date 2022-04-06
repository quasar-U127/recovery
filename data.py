from glob import escape, glob
import os
from torch.utils.data import Dataset
import numpy as np
from typing import Any, Tuple, List


class UndoData(Dataset):
    def __init__(self, path: str):
        self.root = path
        self.samples = [os.path.splitext(os.path.basename(f))[0] for f in glob(
            os.path.join(escape(self.root), "*.npz"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = self.samples[index]
        sample = np.load(os.path.join(self.root, f"{item}.npz"))
        return sample["original"], sample["altered"], sample["undo"]


class AlterationData(Dataset):

    @staticmethod
    def extract_sample(path: str):
        path, _ = os.path.splitext(path)
        path, sample = os.path.split(path)
        _, subset = os.path.split(path)
        return subset, sample

    def __init__(self, path: str, subsets: List[str]):
        self.root = path
        self.samples: List[Tuple[str, str]] = []

        if subsets is None or len(subsets)==0:
            self.samples = [self.extract_sample(f) for f in glob(
                os.path.join(escape(self.root), "*", "*.npz"))]
        else:
            for subset in subsets:
                self.samples += [
                    self.extract_sample(f)
                    for f in glob(os.path.join(escape(self.root), escape(subset), "*.npz"))]

    def get_att_names(self):
        return ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        subset, item = self.samples[index]
        sample = np.load(os.path.join(self.root, subset, f"{item}.npz"))
        return sample["original"], sample["recontruction"], sample["altered"], sample["change"]
