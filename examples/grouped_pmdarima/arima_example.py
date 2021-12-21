from pmdarima.arima.arima import ARIMA
from examples.example_data_generator import generate_example_data
from diviner import GroupedPmdarima


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

    # Build a GroupedPmdarima model by specifying an ARIMA model
    arima_obj = ARIMA(order=(2, 1, 3), out_of_sample_size=60)
    base_arima = GroupedPmdarima(
        y_col="y", datetime_col="ds", model_template=arima_obj
    ).fit(df=training_data, group_key_columns=group_key_columns, silence_warnings=True)

    # Save to local directory
    save_dir = "/tmp/group_pmdarima/arima.gpmd"
    base_arima.save(save_dir)

    # Load from saved model
    loaded_model = GroupedPmdarima.load(save_dir)

    print("\nARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(loaded_model)

    prediction = loaded_model.predict(n_periods=30, alpha=0.02, return_conf_int=True)
    print("\nPredictions:\n", "-" * 40)
    print(prediction.to_string())
