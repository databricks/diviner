import warnings
import numpy as np
from pmdarima.model_selection import cross_val_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def _cross_validate_single_model(
    model, group_series, metrics, cross_validator, error_score, exog, verbosity
):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if isinstance(metrics, str):
            metrics = list(metrics)

        output = {}
        for metric in metrics:
            scores = cross_val_score(
                estimator=model,
                y=group_series,
                X=exog,
                scoring=metric,
                cv=cross_validator,
                verbose=verbosity,
                error_score=error_score,
            )
            output[f"{metric}_mean"] = np.mean(scores)
            output[f"{metric}_stddev"] = np.std(scores)

    return output


def _cross_validate_grouped_pmdarima(
    grouped_model,
    grouped_data,
    y_col,
    metrics,
    cross_validator,
    error_score,
    exog_cols=None,
    verbosity=0,
):

    output = {}
    for group, data in grouped_data:
        exog = data[exog_cols] if exog_cols else None
        output[group] = _cross_validate_single_model(
            model=grouped_model.get(group),
            group_series=data[y_col],
            metrics=metrics,
            cross_validator=cross_validator,
            error_score=error_score,
            exog=exog,
            verbosity=verbosity,
        )
    return output
