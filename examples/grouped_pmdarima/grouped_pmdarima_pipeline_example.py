import numpy as np
from pmdarima.arima.auto import AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.model_selection import RollingForecastCV
from diviner.utils.example_utils.example_data_generator import generate_example_data
from diviner import GroupedPmdarima


def get_and_print_model_metrics_params(grouped_model):
    fit_metrics = grouped_model.get_metrics()
    fit_params = grouped_model.get_model_params()

    print(f"\nModel Fit Metrics:\n{fit_metrics.to_string()}")
    print(f"\nModel Fit Params:\n{fit_params.to_string()}")


if __name__ == "__main__":

    # Generate a few years of daily data across 2 different groups, defined by 3 columns that
    # define each group
    generated_data = generate_example_data(
        column_count=3,
        series_count=2,
        series_size=365 * 3,
        start_dt="2019-01-01",
        days_period=1,
    )

    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    pipeline_obj = Pipeline(
        steps=[
            (
                "box",
                BoxCoxEndogTransformer(lmbda2=0.4, neg_action="raise", floor=1e-12),
            ),
            ("arima", AutoARIMA(out_of_sample_size=60, max_p=4, max_q=4, max_d=4)),
        ]
    )
    pipeline_arima = GroupedPmdarima(model_template=pipeline_obj).fit(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
        silence_warnings=True,
    )

    # Save to local directory
    save_dir = "/tmp/group_pmdarima/pipeline.gpmd"
    pipeline_arima.save(save_dir)

    # Load from saved model
    loaded_model = GroupedPmdarima.load(save_dir)

    print("\nPipeline AutoARIMA results:\n", "-" * 40)
    get_and_print_model_metrics_params(loaded_model)

    print("\nPredictions:\n", "-" * 40)
    prediction = loaded_model.predict(
        n_periods=30, alpha=0.2, predict_col="predictions", return_conf_int=True
    )
    print(prediction.to_string())

    print("\nCross validation metric results:\n", "-" * 40)
    cross_validator = RollingForecastCV(h=30, step=365, initial=730)
    cv_results = loaded_model.cross_validate(
        df=training_data,
        metrics=["mean_squared_error"],
        cross_validator=cross_validator,
        error_score=np.nan,
        verbosity=3,
    )

    print(cv_results.to_string())
