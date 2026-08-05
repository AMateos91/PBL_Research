from __future__ import annotations

import numpy as np

from PBL_Research.ml.neural.mlp import MLPModel
from PBL_Research.ml.explainability import (
    PermutationImportance,
    SHAPExplainer,
)


def main() -> None:

    np.random.seed(42)

    input_size = 10

    model = MLPModel(
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=1,
    )

    x = np.random.randn(
        100,
        input_size,
    )

    y = np.random.randn(
        100,
    )

    print("=== Permutation Importance ===")

    permutation = PermutationImportance(
        model,
    )

    permutation_values = permutation.explain(
        x,
        y,
    )

    print(permutation_values)

    print("\n=== SHAP ===")

    background = x[:20]

    shap = SHAPExplainer(
        model,
        background,
    )

    shap_values = shap.explain(
        x,
    )

    print(shap_values)


if __name__ == "__main__":

    main()
