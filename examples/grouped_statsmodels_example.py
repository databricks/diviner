from examples.example_data_generator import generate_example_data
from diviner import GroupedStatsmodels

if __name__ == "__main__":
    # This function call will generate synthetic group time series data in a normalized format.
    # The structure will be of:
    # |ds |y |group_key_1 | group_key_2| group_key_3 |
    # With the grouping key values that are generated per ds and y values assigned in a
    # non-deterministic fashion.

    # For utililzation of this API, the normalized representation of the data is required, such that
    # a particular target variables' data 'y' and the associated indexed datetime values in 'ds' are
    # 'stacked' (unioned) from a more traditional denormalized data storage paradigm.

    # For guidance on this data transposition from denormalized representations, see:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
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
    grouping_key_columns = generated_data.key_columns

    # Create a GroupedStatsmodels ExponentialSmoothing model instance
    grouped_model = GroupedStatsmodels(
        model_type="ExponentialSmoothing", endog="y", time_col="ds"
    ).fit(training_data, grouping_key_columns)

    # Save the model to the local file system
    save_path = "/tmp/grouped_exponential_smoothing.gsm"
    grouped_model.save(path=save_path)

    # Load the model from the local storage location
    retrieved_model = GroupedStatsmodels.load(save_path)

    # Score the model and print the results
    model_scores = retrieved_model.get_metrics()

    print(f"Model scores:\n{model_scores.to_string()}")

    # Run a forecast for each group
    forecasts = retrieved_model.forecast(horizon=20, frequency="D")

    print(f"Forecasted data:\n{forecasts[:50].to_string()}")

    # Extract the parameters from each model for logging
    params = retrieved_model.get_model_params()

    print(f"Model parameters:\n{params.to_string()}")
