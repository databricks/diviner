import inspect

from prophet import Prophet


def _get_extract_params():
    """
    Utility function for extracting the parameters from a Prophet model.
    The changepoints attribute is being removed due to the fact that it is a utility NumPy array
    defining the detected changepoint indexes (datetime values) from the input training data set.
    There is no utility to recording this for parameter adjustment or historical reference as it
    is used exclusively internally by Prophet.

    :return: A list of parameters to extract from a Prophet model for model tracking purposes.
    """
    blacklist = {
        "changepoints",
        "changepoints_t",
        "seasonalities",
        "stan_fit",
        "stan_backend",
        "params",
        "history",
        "history_dates",
        "train_component_cols",
    }
    prophet_signature = [
        attr
        for attr, value in inspect.getmembers(Prophet())
        if not callable(value) and not attr.startswith("__") and not attr in blacklist
    ]

    return prophet_signature
