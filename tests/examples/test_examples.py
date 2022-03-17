import os
import pytest
from ..tools import process

EXAMPLES_DIR = "./examples"


@pytest.mark.parametrize(
    "directory, command",
    [
        ("grouped_pmdarima", ["python", "grouped_pmdarima_arima_example.py"]),
        ("grouped_pmdarima", ["python", "grouped_pmdarima_autoarima_example.py"]),
        ("grouped_pmdarima", ["python", "grouped_pmdarima_series_exploration.py"]),
        ("grouped_pmdarima", ["python", "grouped_pmdarima_pipeline_example.py"]),
        (
            "grouped_pmdarima",
            ["python", "grouped_pmdarima_subset_prediction_example.py"],
        ),
        (
            "grouped_pmdarima",
            ["python", "grouped_pmdarima_analyze_differencing_terms_and_apply.py"],
        ),
        ("grouped_prophet", ["python", "grouped_prophet_example.py"]),
        ("grouped_prophet", ["python", "grouped_prophet_subset_prediction_example.py"]),
    ],
)
def test_examples(directory, command):
    script_directory = os.path.join(EXAMPLES_DIR, directory)
    process.exec_cmd(command, cwd=script_directory)
