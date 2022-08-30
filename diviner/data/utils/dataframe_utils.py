import pandas as pd


def _apply_datetime_index(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Application of datetime index to a single DataFrame to support format requirements for
    certain timeseries models.

    :param df: An individual group's DataFrame
    :param datetime_col: The column that defines the datetime data for the series
    :return: An individual group's series with the temporal component encoded as the index.
    """
    frequency = pd.infer_freq(df[datetime_col])
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    df.index.freq = frequency

    return df


def apply_datetime_index_to_groups(grouped_data, datetime_col: str):
    """
    Utility function to strip the datetime column from the submitted dataframe,
    infer the frequency, and apply this as the DataFrame index, preserving the frequency data.

    :param grouped_data: The grouped structure from
                         `PandasGroupGenerator.generate_processing_groups`, consisting of a
                         collection of tuples of (master_group_key, DataFrame)
    :param datetime_col: The column name of the source user-input DataFrame that defines the
                         datetime values of the series
    :return: A DatetimeIndex-applied collection of (master_group_key, DataFrame) from the original
             group processed collection wherein the DataFrame for each group has had its index
             set by the inferred frequency of the datetime_col column.
    """
    datetime_indexed = [
        (master_key, _apply_datetime_index(df, datetime_col)) for master_key, df in grouped_data
    ]

    return datetime_indexed
