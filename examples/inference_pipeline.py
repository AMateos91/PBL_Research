from __future__ import annotations

import torch

from PBL_Research.ml.neural.mlp import MLPModel
from PBL_Research.ml.inference import Predictor


def main() -> None:

    torch.manual_seed(42)

    input_size = 10

    model = MLPModel(
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=1,
    )

    predictor = Predictor(
        model=model,
    )

    x = torch.randn(
        5,
        input_size,
    )

    predictions = predictor.predict(
        x,
    )

    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
