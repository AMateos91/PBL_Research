from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class DataLoaderBuilder:
    """
    Builder for PyTorch DataLoader objects.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def build(
        self,
    ) -> DataLoader:
        """
        Build a PyTorch DataLoader.
        """

        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
