import numpy as np
from pmdarima.arima.arima import ARIMA
from pmdarima.model_selection import SlidingWindowForecastCV
from diviner.utils.example_utils.example_data_generator import generate_example_data
from diviner import GroupedPmdarima


def get_and_print_model_metrics_params(grouped_model):
    fit_metrics = grouped_model.get_metrics()
    fit_params = grouped_model.get_model_params()

    print(f"\nModel Fit Metrics:\n{fit_metrics.to_string()}")
    print(f"\nModel Fit Params:\n{fit_params.to_string()}")


if __name__ == "__main__":

    # Generate a few years of daily data across 4 different groups, defined by 3 columns that
    # define each group
    generated_data = generate_example_data(
        column_count=3,
        series_count=4,
        series_size=365 * 4,
        start_dt="2019-01-01",
        days_period=1,
    )

    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    # Build a GroupedPmdarima model by specifying an ARIMA model
    arima_obj = ARIMA(order=(2, 1, 3), out_of_sample_size=60)
    base_arima = GroupedPmdarima(model_template=arima_obj).fit(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
        silence_warnings=True,
    )

    # Save to local directory
    save_dir = "/tmp/group_pmdarima/arima.gpmd"
    base_arima.save(save_dir)

    # Load from saved model
    loaded_model = GroupedPmdarima.load(save_dir)

    print("\nARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(loaded_model)

    prediction = loaded_model.predict(
        n_periods=30, alpha=0.02, predict_col="forecast", return_conf_int=True
    )
    print("\nPredictions:\n", "-" * 40)
    print(prediction.to_string())

    print("\nCross validation metric results:\n", "-" * 40)
    cross_validator = SlidingWindowForecastCV(h=90, step=365, window_size=730)
    cv_results = loaded_model.cross_validate(
        df=training_data,
        metrics=["mean_squared_error", "smape", "mean_absolute_error"],
        cross_validator=cross_validator,
        error_score=np.nan,
        verbosity=4,
    )

    print(cv_results.to_string())
