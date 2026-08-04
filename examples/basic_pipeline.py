"""
examples/basic_pipeline.py

Minimal end-to-end example using PBL_Research.
"""

from PBL_Research.acquisition import EarthData
from PBL_Research.datasets import (
    DatasetBuilder,
    Selection,
    Target,
    Split,
    Scaling,
    Tensor,
    DataLoaderBuilder,
)
from PBL_Research.ml import Model
from PBL_Research.workflows import Pipeline


def main():

    # -------------------------------------------------
    # 1. Data acquisition
    # -------------------------------------------------
    acquisition = EarthData()

    dataset = acquisition.load(
        source="sample_dataset"
    )

    # -------------------------------------------------
    # 2. Dataset preparation
    # -------------------------------------------------
    builder = DatasetBuilder(dataset)

    builder.apply(Selection())
    builder.apply(Target("target"))
    builder.apply(Split(test_size=0.2))
    builder.apply(Scaling())
    builder.apply(Tensor())

    train_loader, test_loader = DataLoaderBuilder(builder).build()

    # -------------------------------------------------
    # 3. Model
    # -------------------------------------------------
    model = Model()

    # -------------------------------------------------
    # 4. Pipeline
    # -------------------------------------------------
    pipeline = Pipeline(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    pipeline.fit()
    results = pipeline.evaluate()

    print(results)


if __name__ == "__main__":
    main()
