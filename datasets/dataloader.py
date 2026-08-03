from __future__ import annotations

import torch


class DataLoaderBuilder:

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def build(
        self,
    ) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
