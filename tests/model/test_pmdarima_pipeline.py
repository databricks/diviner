import pytest
from pmdarima.arima import AutoARIMA
from diviner.model.pmdarima_pipeline import PmdarimaPipeline
from diviner.exceptions import DivinerException


def test_base_pipeline_construction_no_overrides():

    pipeline = PmdarimaPipeline()._build_pipeline_stages()
    first_stage, first_class = pipeline[0]
    assert len(pipeline) == 1
    assert first_stage == "arima"
    assert isinstance(first_class, AutoARIMA)


def test_multiple_overrides():

    conf = {
        "fourier__m": 6,
        "fourier__k": 4,
        "boxcox__lmbda": 0.1,
        "log__lmbda": 200,
        "arima__alpha": 0.1,
    }
    pipeline = PmdarimaPipeline(**conf)._build_pipeline_stages()

    fourier_stage, fourier_class = pipeline[0]
    boxcox_stage, boxcox_class = pipeline[1]
    log_stage, log_class = pipeline[2]
    arima_stage, arima_class = pipeline[3]

    assert fourier_stage == "fourier"
    assert boxcox_stage == "boxcox"
    assert log_stage == "log"
    assert arima_stage == "arima"

    assert getattr(fourier_class, "m") == 6
    assert getattr(fourier_class, "k") == 4
    assert getattr(boxcox_class, "lmbda") == 0.1
    assert getattr(log_class, "lmbda2") == 200  # Internally gets set as "lmbda2"
    assert log_class.get_params(deep=False)["lmbda"] == 200
    assert getattr(arima_class, "alpha") == 0.1


def test_bad_format_of_params_raises():

    with pytest.raises(
        DivinerException,
        match="Error in extracting configurations for stage "
        "'arima'. Ensure that stage and argument are "
        "separated by 2 underscores.",
    ):
        PmdarimaPipeline(arima_alpha=0.01)._build_pipeline_stages()


def test_invalid_params_raises():

    with pytest.raises(
        TypeError,
        match="__init__\\(\\) got an unexpected keyword argument 'invalid'",
    ):
        PmdarimaPipeline(
            fourier__m=8, fourier__invalid="invalid"
        )._build_pipeline_stages()


def test_missing_required_params_raises():

    with pytest.raises(
        DivinerException,
        match="Using a FourierFeaturizer requires the key 'fourier__m' to be set. "
        "Keys passed: fourier__k",
    ):
        PmdarimaPipeline(fourier__k=2)._build_pipeline_stages()

    with pytest.raises(
        DivinerException,
        match="Using a DateFeaturizer requires the key 'dates__column_name' to be set. "
        "Keys passed: dates",
    ):
        PmdarimaPipeline(dates=True)._build_pipeline_stages()


def test_pipeline_build_by_stage_order_definition():

    conf = {
        "stage_order": ["log", "boxcox", "fourier", "dates"],
        "fourier__m": 2,
        "dates__column_name": "dt",
    }
    pipeline = PmdarimaPipeline(**conf).build_pipeline()
    steps = pipeline.steps
    for i, value in enumerate(conf["stage_order"]):
        assert steps[i][0] == value
    assert steps[-1][0] == "arima"


def test_pipeline_order_invalid_arima_placement_raises():

    bad_conf = {"stage_order": ["arima", "log", "boxcox"]}

    with pytest.raises(
        DivinerException,
        match="Ordered preprocessing stages submitted to pipeline must have 'arima' as final "
        "stage. Submitted order: 'arima, log, boxcox'",
    ):
        PmdarimaPipeline(**bad_conf).build_pipeline()


def test_pipeline_builds_incomplete_ordering():

    conf = {
        "stage_order": ["dates", "boxcox"],
        "fourier__m": 4,
        "arima__max_p": 6,
        "dates__column_name": "dt",
    }

    pipeline = PmdarimaPipeline(**conf).build_pipeline()
    steps = pipeline.steps
    step_names = [step for step, step_class in steps]
    assert step_names == ["dates", "boxcox", "fourier", "arima"]
