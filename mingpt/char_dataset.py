from dataclasses import dataclass
from typing import Optional

import fsspec
import torch
from torch.utils.data import Dataset
"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""


@dataclass
class DataConfig:
    path: Optional[str] = None
    block_size: Optional[int] = None
    train_split: Optional[float] = None
    truncate: float = 1.0


class CharDataset(Dataset):

    def __init__(self, config: DataConfig):
        with fsspec.open(config.path) as f:
            data = f.read()
        data = data[:int(len(data) * config.truncate)]

        chars = sorted(list(set(data)))
        print(f"Data has {len(data)} chars and {len(chars)} unique characters")
        self.stoi = {char: idx for idx, char in enumerate(chars)}
        self.itos = {idx: char for idx, char in enumerate(chars)}
        self.block_size = config.block_size
        self.vocab_size = len(chars)
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # We want to grab chars in blocks
        block = self.data[idx:idx + self.block_size + 1]
        # Encode to chars to integers
        indices = [self.stoi[char] for char in block]
        # Grab everything but last char in block
        inputs = torch.tensor(indices[:-1], dtype=torch.long)
        # Grab everything but first char in block
        labels = torch.tensor(indices[1:], dtype=torch.long)
        return inputs, labels
