from pmdarima.arima.arima import ARIMA
from diviner import GroupedPmdarima
from diviner.utils.example_utils.example_data_generator import generate_example_data

if __name__ == "__main__":

    generated_data = generate_example_data(
        column_count=2,
        series_count=6,
        series_size=365 * 4,
        start_dt="2019-01-01",
        days_period=1,
    )

    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    arima_obj = ARIMA(order=(2, 1, 3), out_of_sample_size=60)
    base_arima = GroupedPmdarima(model_template=arima_obj).fit(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
        silence_warnings=True,
    )

    # Get a subset of group keys to generate forecasts for
    group_df = training_data.copy()
    group_df["groups"] = list(zip(*[group_df[c] for c in group_key_columns]))
    distinct_groups = group_df["groups"].unique()
    groups_to_predict = list(distinct_groups[:3])

    print("-" * 65)
    print(f"Unique groups that have been modeled: {distinct_groups}")
    print(f"Subset of groups to generate predictions for: {groups_to_predict}")
    print("-" * 65)

    forecasts = base_arima.predict_groups(
        groups=groups_to_predict,
        n_periods=60,
        predict_col="forecast_values",
        on_error="warn",
    )

    print(f"\nForecast values:\n{forecasts.to_string()}")
