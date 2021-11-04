from prophet import Prophet
from inspect import signature
from diviner.config.grouped_prophet.utils import prophet_config_utils


def get_scoring_metrics(user_metrics=None):
    """
    Utility function for validating user-supplied metrics or, if not supplied, return the default
    metrics for this module: `("mse", "rmse", "mae", "mape", "mdape", "smape", "coverage")`

    The collection of scoring metrics defined are the base implementations from Prophet's
    `prophet.diagnostics` module and are set and defined as the default metrics to be utilized for
    calculation through the `prophet.diagnostics.performance_metrics()` function.
    For reference:
    mse: Mean Squared Error - a representative value of the squared error between predicted and
         actual values, averaged over each observation point.
         https://en.wikipedia.org/wiki/Mean_squared_error
    rmse: Root Mean Squared Error - the square root of the mse. This is a useful metric to reduce
          the comparative extreme of a small number of extreme outlier points dominating the
          metric of mse.
          https://en.wikipedia.org/wiki/Root-mean-square_deviation
    mae: Mean Absolute Error - an alternative approach to rmse; the average of the absolute values
         of the error term of the difference between actual and predicted for each observation.
         https://en.wikipedia.org/wiki/Mean_absolute_error
    mape: Mean Absolute Percentage Error - A scaled measurement of the difference of each predicted
          value to the actual divided by the actual value. This percentage allows for a more
          constrained metric space than other alterantives (aiding legibility), but can be highly
          misleading if there are values of '0' contained within the observed space (the
          denominator either needs adjustment or filtering of that entry's value)
          https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    mdape: Median Absolute Percentage Error - similar to mape, but utilizing a median value
           instead of an average to resolve a single value for the errors of the series.
    smape: Symmetric Mean Absolute Percentage Error - An evolution of the mape algorithm that,
           instead of dividing by the actual value in the demominator as in mape, divides by the
           average of the actual and predicted values. This implementation in Prophet uses the
          Chen and Yang formula from 2004.
          https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    coverage: The percentage of predicted values that fall within the error estimations (yhat_upper,
              yhat_lower) from the error estimates derived from backtesting estimation. This metric
              is only calculated if error estimates are configured (they are on by default) in the
              initialization of a Prophet model. To disable them (if this metric is not needed), set
              the argument `uncertainty_samples=0` in the class constructor.
    :param user_metrics: an iterable collection of metric names that must be a subset of the
                         default available metrics.
    :return:
    """
    if user_metrics:
        return prophet_config_utils._validate_user_metrics(user_metrics)
    else:
        return prophet_config_utils._SCORING_METRICS


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



