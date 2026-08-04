from pbl_research.evaluation.metrics import Metrics


def test_accuracy():

    value = Metrics.accuracy(

        [0, 1, 1],

        [0, 1, 0],

    )

    assert 0 <= value <= 1
