import numpy as np

from pbl_research.ml.classical.linear import LinearRegressionModel


def test_linear_model():

    x = np.random.rand(50, 4)

    y = np.random.rand(50)

    model = LinearRegressionModel()

    model.fit(
        x,
        y,
    )

    prediction = model.predict(
        x,
    )

    assert prediction.shape[0] == x.shape[0]
