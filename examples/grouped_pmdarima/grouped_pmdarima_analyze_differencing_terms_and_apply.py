from diviner.utils.example_utils.example_data_generator import generate_example_data
from diviner import GroupedPmdarima, PmdarimaAnalyzer
from pmdarima.pipeline import Pipeline
from pmdarima import AutoARIMA
from pmdarima.model_selection import SlidingWindowForecastCV


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
        series_count=3,
        series_size=365 * 3,
        start_dt="2019-01-01",
        days_period=1,
    )

    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    pipeline = Pipeline(
        steps=[
            (
                "arima",
                AutoARIMA(
                    max_order=14,
                    out_of_sample_size=90,
                    suppress_warnings=True,
                    error_action="ignore",
                ),
            )
        ]
    )

    diff_analyzer = PmdarimaAnalyzer(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
    )
    ndiff = diff_analyzer.calculate_ndiffs(
        alpha=0.05,
        test="kpss",
        max_d=4,
    )

    grouped_model = GroupedPmdarima(model_template=pipeline).fit(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
        ndiffs=ndiff,
        silence_warnings=True,
    )

    # Save to local directory
    save_dir = "/tmp/group_pmdarima/pipeline_override.gpmd"
    grouped_model.save(save_dir)

    # Load from saved model
    loaded_model = GroupedPmdarima.load(save_dir)

    print("\nAutoARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(loaded_model)

    print("\nPredictions:\n", "-" * 40)
    prediction = loaded_model.predict(
        n_periods=30, alpha=0.1, predict_col="forecasted_values", return_conf_int=True
    )
    print(prediction.to_string())

    cv_evaluator = SlidingWindowForecastCV(h=90, step=120, window_size=180)
    cross_validation = loaded_model.cross_validate(
        df=training_data,
        metrics=["smape", "mean_squared_error", "mean_absolute_error"],
        cross_validator=cv_evaluator,
    )

    print("\nCross validation metrics:\n", "-" * 40)
    print(cross_validation.to_string())
