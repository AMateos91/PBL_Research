from .base import NeuralModel

from .mlp import MLPModel
from .cnn import CNNModel

__all__ = [
    "NeuralModel",
    "MLPModel",
    "CNNModel",
]
