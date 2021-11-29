import inspect

from prophet import Prophet


def _get_extract_params():
    """
    Utility function for extracting the parameters from a Prophet model.
    The following attributes are not considered for parameter extraction for the reasons listed:
    The changepoints attribute is being removed due to the fact that it is a utility NumPy array
    defining the detected changepoint indexes (datetime values) from the input training data set.
    There is no utility to recording this for parameter adjustment or historical reference as it
    is used exclusively internally by Prophet.
    The changepoints_t attribute is being removed for a similar reason as changepoints is. It is
    a list of changepoints datetime values that are generated during model training and cross
    validation that are outside of the control of the end-user.
    The seasonalities attribute is a collection that is populated during fit of the model and is
    not a parameter that can be adjusted directly by the user.
    stan_fit is a result of the underlying solver's iterations for fitting. It is non-useful for
    collection as a model parameter as it cannot be direclty changed by the user.
    params are additional attributes set during training that are not directly controllable by the
    user.
    history and history_dates are extracted copies of the underlying training data used during
    training of the model and to generate certain plots from within the library.
    train_component_cols are generated during training as part of the trend decomposition. A user
    has no ability to modify this behavior.
    :return: A list of parameters to extract from a Prophet model for model tracking purposes.
    """
    denylist = {
        "changepoints",
        "changepoints_t",
        "seasonalities",
        "stan_fit",
        "params",
        "history",
        "history_dates",
        "train_component_cols",
    }
    prophet_signature = [
        attr
        for attr, value in inspect.getmembers(Prophet())
        if not callable(value) and not attr.startswith("_") and not attr in denylist
    ]

    return prophet_signature
