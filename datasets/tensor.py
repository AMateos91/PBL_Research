from __future__ import annotations

import numpy as np
import torch

from ...utils.constants import Framework


class Tensor:

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        framework: Framework = Framework.PYTORCH,
        dtype: torch.dtype = torch.float32,
    ):

        if framework == Framework.PYTORCH:

            return torch.as_tensor(
                array,
                dtype=dtype,
            )

        if framework == Framework.NUMPY:

            return array

        if framework == Framework.TENSORFLOW:

            try:

                import tensorflow as tf

            except ImportError as error:

                raise ImportError(
                    "TensorFlow is not installed."
                ) from error

            return tf.convert_to_tensor(
                array,
            )

        raise ValueError(
            f"Unsupported framework: {framework.value}"
        )
