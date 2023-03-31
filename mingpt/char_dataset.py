import torch
from torch.utils.data import Dataset
"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""


class CharDataset(Dataset):

    def __init__(self, path: str, block_size: int):
        with open(path, "r") as f:
            data = f.read()

        chars = sorted(list(set(data)))
        print(f"Data has {len(data)} chars and {len(chars)} unique characters")
        self.stoi = {char: idx for idx, char in enumerate(chars)}
        self.itos = {idx: char for idx, char in enumerate(chars)}
        self.block_size = block_size
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
