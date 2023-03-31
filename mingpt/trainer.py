"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
from collections import OrderedDict
import os

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@dataclass
class GPTTrainerConfig:
    job_name: str
    dl_num_workers: int = 4
    max_epochs: int = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: Tuple[float] = (0.9, 0.95)
    grad_norm_clip: float = 0.1
    snapshot_path: Optional[str] = None


# class to hold model state
@dataclass
class ModelSnapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    final_epoch: int


class GPTTrainer:
    def __init__(self, config: GPTTrainerConfig, model: torch.nn.Module, optimizer: Any, train_dataset: Dataset, test_dataset: Dataset=None):
        # Torchrun settings
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.config = config
        self.train_dataset = train_dataset

        # setup the dataloader
        self.train_loader = self._get_dataloader(train_dataset)
        self.test_loader = self._get_dataloader(test_dataset) if test_dataset else None

        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = config.save_every
        # Load model snapshot if available
        self._load_snapshot(config.snapshot_path)
        # Wrap model in DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.dl_num_workers,
            sampler=DistributedSampler(dataset),
        )

    def _load_snapshot(self,path: str):
        if os.path.exists(path):
            # If the model snapshot is already there, load it onto the cpu
            snapshot_data = torch.load(path,map_location="cpu")
            # Load into ModelSnapshot object
            model_snapshot = ModelSnapshot(**snapshot_data)
            # Load model states into model which is on GPU
            self.model.load_state_dict(model_snapshot.model_state)
            # Load optimizer states on GPU
            self.optimizer.load_state_dict(model_snapshot.optimizer_state)
            # Last epoch
            self.last_epoch = model_snapshot.final_epoch
            print(f"Resuming training from epoch: {self.last_epoch}")
        else:
            print("ModelSnapshot not found, training model from scratch!")

    def _run_batch(self, inputs, labels, train: bool = True) -> float:
        """Run forward and backward pass then compute loss"""
        with torch.set_grad_enabled(train):
            # Foward pass to compute loss and activations
            logits, loss = self.model(inputs, labels)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            # Backward pass to compute gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm(
                self.model.parameters(),
                self.config.grap_norm_clip
            )
            # Update parameters
            self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True) -> None:
        """Run epoch for batch of data"""
        for idx, (x, y) in enumerate(dataloader):
            inputs = x.to(self.local_rank)
            labels = y.to(self.local_rank)
            loss = self._run_batch(inputs, labels, train)
            if idx % 100 == 0:
                print(f"[GPU{self.local_rank}] Epoch {epoch+1},{'Training' if train else 'Test'} loss {loss}")

    def _snapshot_model(self, epoch: int) -> None:
        """Checkpoint the model state and optimizer state for a given
        epoch. """
        model = self.model.module if hasattr(self.model,"module") else self.model
        model_snapshot = ModelSnapshot(
            model_state=model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            final_epoch=epoch,
        )
        # Save snapshot as dictionary (it pickles underneath)
        model_snapshot_file = "model_snapshot.pt"
        torch.save(asdict(model_snapshot),model_snapshot_file)
        print(f"Model snapshot taken and saved at epoch {epoch}, filename: {model_snapshot_file}")

    def train(self, max_epochs: int) -> None:
        """Train model for max_epochs. We also snapshot
        the model and optimizer states every 'self.save_every' epochs"""
        # Tricky: Make sure you train starting at the last snapshot epoch, or else you will start
        # from scratch everytime.
        for epoch in range(self.last_epoch, self.max_epochs):
            self._run_epoch(epoch, self.train_loader, True)
            # We only snapshot the model at the head node
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._snapshot_model(epoch)
            # Evaluate on test set if test_loader is available, else its None
            if self.test_loader is not None:
                # Set train to false so we run in evaluation mode and
                # do not compute gradients
                self._run_epoch(epoch, self.test_dataset, False)








