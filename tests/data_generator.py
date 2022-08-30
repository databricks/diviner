import itertools
import pandas as pd
import numpy as np
import string
import random
from datetime import timedelta, datetime
from collections import namedtuple


def _generate_time_series(series_size: int):
    residuals = np.random.lognormal(
        mean=np.random.uniform(low=0.5, high=3.0),
        sigma=np.random.uniform(low=0.6, high=0.98),
        size=series_size,
    )
    trend = [
        np.polyval([15.0, 1.0, 5], x)
        for x in np.linspace(start=0, stop=np.random.randint(low=0, high=4), num=series_size)
    ]
    seasonality = [
        30 * np.sin(2 * np.pi * 1000 * (i / (series_size * 200))) + 40
        for i in np.arange(0, series_size)
    ]

    return residuals + trend + seasonality + np.random.uniform(low=20.0, high=1000.0)


def _generate_grouping_columns(column_count: int, series_count: int):
    candidate_list = list(string.ascii_uppercase)
    candidates = random.sample(
        list(itertools.permutations(candidate_list, column_count)), series_count
    )
    column_names = sorted([f"key{x}" for x in range(column_count)], reverse=True)
    return [dict(zip(column_names, entries)) for entries in candidates]


def _generate_raw_df(
    column_count: int,
    series_count: int,
    series_size: int,
    start_dt: str,
    days_period: int,
):
    candidates = _generate_grouping_columns(column_count, series_count)
    start_date = datetime.strptime(start_dt, "%Y-%M-%d")
    dates = np.arange(
        start_date,
        start_date + timedelta(days=series_size * days_period),
        timedelta(days=days_period),
    )
    df_collection = []
    for entry in candidates:
        generated_series = _generate_time_series(series_size)
        series_dict = {"ds": dates, "y": generated_series}
        series_df = pd.DataFrame.from_dict(series_dict)
        for column, value in entry.items():
            series_df[column] = value
        df_collection.append(series_df)
    return pd.concat(df_collection)


def generate_test_data(
    column_count: int,
    series_count: int,
    series_size: int,
    start_dt: str,
    days_period: int = 1,
):

    Structure = namedtuple("Structure", "df key_columns")
    data = _generate_raw_df(column_count, series_count, series_size, start_dt, days_period)
    key_columns = list(data.columns)

    for key in ["ds", "y"]:
        key_columns.remove(key)

    return Structure(data, key_columns)
