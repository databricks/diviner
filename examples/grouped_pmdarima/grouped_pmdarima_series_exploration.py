import pprint
from diviner import PmdarimaAnalyzer

from diviner.utils.example_utils.example_data_generator import generate_example_data


def _print_dict(data, name):
    print("\n" + "-" * 100)
    print(f"{name} values for the groups")
    print("-" * 100, "\n")
    pprint.PrettyPrinter(indent=2).pprint(data)


if __name__ == "__main__":

    generated_data = generate_example_data(
        column_count=4,
        series_count=3,
        series_size=365 * 12,
        start_dt="2010-01-01",
        days_period=1,
    )
    training_data = generated_data.df
    group_key_columns = generated_data.key_columns

    # Create a utility object for performing analyses
    # We reuse this object because the grouped data set collection is lazily evaluated and can be
    # reused for subsequent analytics operations on the data set.
    analyzer = PmdarimaAnalyzer(
        df=training_data,
        group_key_columns=group_key_columns,
        y_col="y",
        datetime_col="ds",
    )

    # Decompose the trends of each group
    decomposed_trends = analyzer.decompose_groups(m=7, type_="additive")

    print("Decomposed trend data for the groups")
    print("-" * 100, "\n")
    print(decomposed_trends[:50].to_string())

    # Calculate optimal differencing for ARMA terms
    ndiffs = analyzer.calculate_ndiffs(alpha=0.1, test="kpss", max_d=5)

    _print_dict(ndiffs, "Differencing")

    # Calculate seasonal differencing
    nsdiffs = analyzer.calculate_nsdiffs(m=365, test="ocsb", max_D=5)

    _print_dict(nsdiffs, "Seasonal Differencing")

    # Get the autocorrelation function for each group
    group_acf = analyzer.calculate_acf(
        unbiased=True, nlags=120, qstat=True, fft=True, alpha=0.05, adjusted=True
    )

    _print_dict(group_acf, "Autocorrelation function")

    # Get the partial autocorrelation function for each group
    group_pacf = analyzer.calculate_pacf(nlags=120, method="yw", alpha=0.05)

    _print_dict(group_pacf, "Partial Autocorrelation function")

    # Perform a diff operation on each group
    group_diff = analyzer.generate_diff(lag=7, differences=1)

    _print_dict(group_diff, "Differencing")

    # Invert the diff operation on each group
    group_diff_inv = analyzer.generate_diff_inversion(
        group_diff, lag=7, differences=1, recenter=True
    )

    _print_dict(group_diff_inv, "Differencing Inversion")
