from pmdarima.arima.auto import AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
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

    pipeline_obj = Pipeline(
        steps=[
            (
                "box",
                BoxCoxEndogTransformer(lmbda2=0.2, neg_action="raise", floor=1e-12),
            ),
            ("arima", AutoARIMA(out_of_sample_size=60, max_p=2, max_q=3, max_d=1)),
        ]
    )
    pipeline_arima = GroupedPmdarima(
        y_col="y", datetime_col="ds", model_template=pipeline_obj
    ).fit(df=training_data, group_key_columns=group_key_columns, silence_warnings=True)

    print("\nPipeline AutoARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(pipeline_arima)

    print("\nPredictions:\n", "-" * 40)
    prediction = pipeline_arima.predict(n_periods=30, alpha=0.2, return_conf_int=True)
    print(prediction.to_string())
