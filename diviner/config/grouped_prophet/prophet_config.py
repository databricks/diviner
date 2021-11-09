from prophet import Prophet
from inspect import signature


def _get_extract_params():
    """
    Utility function for extracting the parameters from a Prophet model.
    The changepoints attribute is being removed due to the fact that it is a utility NumPy array
    defining the detected changepoint indexes (datetime values) from the input training data set.
    There is no utility to recording this for parameter adjustment or historical reference as it
    is used exclusively internally by Prophet.

    :return: A list of parameters to extract from a Prophet model for model tracking purposes.
    """
    prophet_signature = list(signature(Prophet).parameters.keys())
    prophet_signature.remove("changepoints")

    return prophet_signature
