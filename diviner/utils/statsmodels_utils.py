def _get_max_datetime_per_group(dt_indexed_group_data):

    continuation_datetimes = {
        group: df.index.max() for group, df in dt_indexed_group_data
    }

    return continuation_datetimes
