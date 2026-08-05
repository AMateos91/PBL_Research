from __future__ import annotations

import numpy as np

from PBL_Research.evaluation import RegressionEvaluator


def main() -> None:

    y_true = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    y_pred = np.array([
        0.9,
        2.1,
        2.8,
        4.2,
        4.9,
    ])

    evaluator = RegressionEvaluator()

    metrics = evaluator.evaluate(
        y_true,
        y_pred,
    )

    print("Evaluation metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
