from __future__ import annotations

import numpy as np
import torch


class Tensor:

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:

        return torch.as_tensor(
            array,
            dtype=dtype,
        )
