from pbl_research.workflows.pipeline import Pipeline


def test_pipeline():

    pipeline = Pipeline([])

    assert pipeline is not None
