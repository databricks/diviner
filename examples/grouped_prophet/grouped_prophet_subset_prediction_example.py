from diviner.utils.example_utils.example_data_generator import generate_example_data
from diviner import GroupedProphet

if __name__ == "__main__":

    generated_data = generate_example_data(
        column_count=3,
        series_count=10,
        series_size=365 * 5,
        start_dt="2016-02-01",
        days_period=1,
    )

    # Extract the normalized grouped datetime series data
    training_data = generated_data.df

    # Extract the names of the grouping columns that define the unique series data
    group_key_columns = generated_data.key_columns

    # Create a GroupedProphet model instance
    grouped_model = GroupedProphet(n_changepoints=20, uncertainty_samples=0).fit(
        training_data, group_key_columns
    )

    # Get a subset of group keys to generate forecasts for
    group_df = training_data.copy()
    group_df["groups"] = list(zip(*[group_df[c] for c in group_key_columns]))
    distinct_groups = group_df["groups"].unique()
    groups_to_predict = list(distinct_groups[:3])

    print("-" * 65)
    print(f"\nUnique groups that have been modeled: \n{distinct_groups}\n")
    print(f"Subset of groups to generate predictions for: \n{groups_to_predict}\n")
    print("-" * 65)

    forecasts = grouped_model.predict_groups(
        groups=groups_to_predict,
        horizon=60,
        frequency="D",
        predict_col="forecast_values",
        on_error="warn",
    )

    print(f"\nForecast values:\n{forecasts.to_string()}")
