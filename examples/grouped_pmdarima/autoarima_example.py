from pmdarima.arima.auto import AutoARIMA
from examples.example_data_generator import generate_example_data
from diviner import GroupedPmdarima
from diviner.utils.pmdarima_utils import generate_prediction_config


def get_and_print_model_metrics_params(grouped_model):
    fit_metrics = grouped_model.get_metrics()
    fit_params = grouped_model.get_model_params()

    print(f"\nModel Fit Metrics:\n{fit_metrics.to_string()}")
    print(f"\nModel Fit Params:\n{fit_params.to_string()}")


if __name__ == "__main__":

    # Generate 6 years of daily data across 4 different groups, defined by 3 columns that
    # define each group
    generated_data = generate_example_data(
        column_count=3,
        series_count=4,
        series_size=365 * 6,
        start_dt="2019-01-01",
        days_period=1,
    )

    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    auto_arima_obj = AutoARIMA(out_of_sample_size=60, maxiter=100)
    base_auto_arima = GroupedPmdarima(
        y_col="y", time_col="ds", model_constructor=auto_arima_obj
    ).fit(df=training_data, group_key_columns=group_key_columns, silence_warnings=True)

    print("\nAutoARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(base_auto_arima)

    pred_conf = generate_prediction_config(
        base_auto_arima,
        n_periods=30,
        alpha=0.1,
        return_conf_int=True,
    )

    print("\nPredictions:\n", "-" * 40)
    prediction = base_auto_arima.predict(pred_conf)
    print(prediction.to_string())

    # Alternatively, the forecast method can be used instead of generating a per-group custom config
    forecast = base_auto_arima.forecast(30, alpha=0.05, return_conf_int=True)
    print("\nForecasts: \n", "-" * 40)
    print(forecast.to_string())
