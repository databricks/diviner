import warnings

import pandas as pd
from packaging.version import Version
from pmdarima.arima import decompose, ndiffs, nsdiffs, is_constant
from pmdarima.utils import acf, pacf, diff, diff_inv
from diviner.data.pandas_group_generator import PandasGroupGenerator
from diviner.utils.common import _restructure_predictions
from diviner.exceptions import DivinerException
from diviner.utils.annotators import experimental


class PmdarimaAnalyzer:
    def __init__(self, df, group_key_columns, y_col, datetime_col):
        """
        A class for performing analysis of a grouped timeseries data set.
        Included in this class are methods for:
        ``decompose_groups``: trend decomposition into `'trend'`, `'seasonal'` and `'random'`
                              elements that collectively make up the underlying series data.
        ``calculate_ndiffs``: optimal selection of the ``d`` (differencing) term for ARIMA to
                              convert a series to one that is stationary. Specifying this as a
                              constant to ``AutoARIMA`` can drastically reduce training time.
        ``calculate_nsdiffs``: optimal selection of the ``D`` (seasonal differencing) term for
                               ``SARIMAX`` to convert a seasonally-influenced series to a stationary
                               one. Providing this as a constant to the ``AutoARIMA`` args
                               ``D=<value>`` can reduce training time, eliminating recursive loops
                               during optimization.
        ``calculate_is_constant``: a validation method to verify that each group's series contains
                                   more than a single value (constancy check). This can aid in
                                   filtering out submitted groups from the ``GroupedPmdarima.fit()``
                                   method to prevent non-useful forecasts from being generated.
                                   (i.e., a repeated constant value throughout a series provides
                                   no value for forecasting and is a waste of runtime resources).
        ``calculate_acf``: grouped calculation of auto correlation factor values for each grouped
                           series.
        ``calculate_pacf``: grouped calculation of partial auto correlation factor values for each
                            grouped series.

        :param df: A DataFrame consisting of at least ``y_col``, ``group_key_columns``, and
                   ``datetime_col`` columns to be analyzed.
        :param group_key_columns: The columns in the ``df`` argument that define, in aggregate, a
                                  unique time series entry.
        :param y_col: The name of the column within the DataFrame that contains the endogenous
                      regressor term.
        :param datetime_col: The name of the column within the DataFrame input that defines the
                             datetime or date values associated with each row of the endogenous
                             regressor ``y_col`` data.
        """
        self._df = df
        self._group_df = None
        self._group_key_columns = group_key_columns
        self._y_col = y_col
        self._datetime_col = datetime_col
        self._master_key = "grouping_key"

    def _create_group_df(self):
        if not self._group_df:
            self._group_df = PandasGroupGenerator(
                self._group_key_columns, self._datetime_col, self._y_col
            ).generate_processing_groups(self._df)

    def _decompose_group(self, group_df, group_key, m, type_, filter_):
        group_df.reset_index(inplace=True)
        group_decomposition = decompose(x=group_df[self._y_col], type_=type_, m=m, filter_=filter_)
        group_result = {
            key: getattr(group_decomposition, key) for key in group_decomposition._fields
        }
        output_df = pd.DataFrame.from_dict(group_result)
        output_df[self._datetime_col] = group_df[self._datetime_col]
        output_df[self._master_key] = output_df.apply(lambda x: group_key, 1)
        return output_df

    @experimental
    def decompose_groups(self, m, type_, filter_=None):
        """
        Utility method that wraps ``pmdarima.arima.decompose()`` for each group within the
        passed-in DataFrame.
        Note: decomposition works best if the total number of entries within the series being
        decomposed is a multiple of the `m` parameter value.

        :param m: The frequency of the endogenous series. (i.e., for daily data, an ``m`` value
                  of ``'7'`` would be appropriate for estimating a weekly seasonality, while
                  setting ``m`` to ``'365'`` would be effective for yearly seasonality effects.)
        :param type_: The type of decomposition to perform.
                      One of: ``['additive', 'multiplicative']``

                      See: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.\
                      decompose.html
        :param filter_: Optional Array for performing convolution. This is specified as a
                        filter for coefficients (the Moving Average and/or
                        Auto Regressor coefficients) in reverse time order in order to filter out
                        a seasonal component.

                        Default: None
        :return: Pandas DataFrame with the decomposed trends for each group.
        """
        self._create_group_df()
        group_decomposition = {
            group_key: self._decompose_group(group_df, group_key, m, type_, filter_)
            for group_key, group_df in self._group_df
        }
        return _restructure_predictions(
            group_decomposition, self._group_key_columns, self._master_key
        )

    @experimental
    def calculate_ndiffs(self, alpha=0.05, test="kpss", max_d=2):
        """
        Utility method for determining the optimal ``d`` value for ARIMA ordering. Calculating this
        as a fixed value can dramatically increase the tuning time for ``pmdarima`` models.

        :param alpha: significance level for determining if a pvalue used for testing a
                      value of ``'d'`` is significant or not.

                      Default: ``0.05``
        :param test: Type of unit test for stationarity determination to use.
                     Supported values: ``['kpss', 'adf', 'pp']``
                     See:

                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.\
                     html#pmdarima.arima.KPSSTest

                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.\
                     html#pmdarima.arima.PPTest

                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.\
                     html#pmdarima.arima.ADFTest

                     Default: ``'kpss'``
        :param max_d: The max value for ``d`` to test.
        :return: Dictionary of ``{<group_key>: <optimal 'd' value>}``
        """
        self._create_group_df()

        group_ndiffs = {
            group: ndiffs(x=group_df[self._y_col], alpha=alpha, test=test, max_d=max_d)
            for group, group_df in self._group_df
        }

        return group_ndiffs

    @experimental
    def calculate_nsdiffs(self, m, test="ocsb", max_D=2):
        """
        Utility method for determining the optimal ``D`` value for seasonal ``SARIMAX`` ordering of
                   ``('P', 'D', 'Q')``.

        :param m: The number of seasonal periods in the series.
        :param test: Type of unit test for seasonality.
                     Supported tests: ``['ocsb', 'ch']``
                     See:

                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.\
                     html#pmdarima.arima.OCSBTest

                     https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.\
                     html#pmdarima.arima.CHTest

                     Default: ``'ocsb'``
        :param max_D: Maximum number of seasonal differences to test for.

                      Default: 2
        :return: Dictionary of ``{<group_key>: <optimal 'D' value>}``
        """
        self._create_group_df()

        group_nsdiffs = {
            group: nsdiffs(x=group_df[self._y_col], m=m, max_D=max_D, test=test)
            for group, group_df in self._group_df
        }

        return group_nsdiffs

    @experimental
    def calculate_is_constant(self):
        """
        Utility method for determining whether or not a series is composed of all of the same
        elements or not. (e.g. a series of {1, 2, 3, 4, 5, 1, 2, 3} will return 'False', while
        a series of {1, 1, 1, 1, 1, 1, 1, 1, 1} will return 'True')

        :return: Dictionary of ``{<group_key>: <Boolean constancy check>}``
        """
        self._create_group_df()
        group_constant_check = {
            group: is_constant(group_df[self._y_col]) for group, group_df in self._group_df
        }
        return group_constant_check

    @experimental
    def calculate_acf(
        self,
        unbiased=False,
        nlags=None,
        qstat=False,
        fft=None,
        alpha=None,
        missing="none",
        adjusted=False,
    ):
        """
        Utility for calculating the autocorrelation function for each group.
        Combined with a partial autocorrelation function calculation, the return values can
        greatly assist in setting AR, MA, or ARMA terms for a given model.

        The general rule to determine whether to use an AR, MA, or ARMA configuration for
        ARIMA (or AutoARIMA) is as follows:

        * ACF gradually trend to significance, PACF significance achieved after 1 lag -> AR model
        * ACF significance after 1 lag, PACF gradually trend to significance -> MA model
        * ACF gradually trend to significance, PACF gradually trend to significance -> ARMA model

        These results can help to set the order terms of an ARIMA model (p and q) or,
        for AutoARIMA, set restrictions on maximum search space terms to assist in faster
        optimization of the model.

        :param unbiased: Boolean flag that sets the autocovariance denominator to ``'n-k'`` if
                         ``True`` and ``n`` if ``False``.

                         Note: This argument is deprecated and removed in versions of pmdarima
                         > 2.0.0

                         Default: ``False``
        :param nlags: The count of autocorrelation lags to calculate and return.

                      Default: ``40``
        :param qstat: Boolean flag to calculate and return the Ljung-Box statistic for each lag.

                      Default: ``False``
        :param fft: Boolean flag for whether to use fast fourier transformation (fft) for
                    computing the autocorrelation function. FFT is recommended for large time
                    series data sets.

                    Default: ``None``
        :param alpha: If specified, calculates and returns the confidence intervals for the
                      acf values at the level set (i.e., for 90% confidence, an alpha of 0.1 would
                      be set)

                      Default: ``None``
        :param missing: handling of NaN values in the series data.

                        Available options:

                        ``['none', 'raise', 'conservative', 'drop']``.

                        ``none``: no checks are performed.

                        ``raise``: an Exception is raised if NaN values are in the series.

                        ``conservative``: the autocovariance is calculated by removing NaN values
                        from the mean and cross-product calculations but are not eliminated from
                        the series.

                        ``drop``: ``NaN`` values are removed from the series and adjacent values
                        to ``NaN``'s are treated as contiguous (which may invalidate the results in
                        certain situations).


                        Default: ``'none'``
        :param adjusted: Deprecation handler for the underlying ``statsmodels`` arguments that have
                         become the ``unbiased`` argument. This is a duplicated value for the
                         denominator mode of calculation for the autocovariance of the series.
        :return: Dictionary of ``{<group_key>: {<acf terms>: <values as array>}}``
        """

        import pmdarima

        self._create_group_df()
        group_acf_data = {}
        for group, df in self._group_df:
            if Version(pmdarima.__version__) < Version("2.0.0"):
                acf_data = acf(  # pylint: disable=unexpected-keyword-arg
                    x=df[self._y_col],
                    unbiased=unbiased,
                    nlags=nlags,
                    qstat=qstat,
                    fft=fft,
                    alpha=alpha,
                    missing=missing,
                )
            else:
                acf_data = acf(
                    x=df[self._y_col],
                    nlags=nlags,
                    qstat=qstat,
                    fft=fft,
                    alpha=alpha,
                    missing=missing,
                    adjusted=adjusted,
                )
            group_data = {"acf": acf_data[0]} if isinstance(acf_data, tuple) else {"acf": acf_data}
            if alpha:
                group_data["confidence_intervals"] = acf_data[1]
                if qstat:
                    group_data["qstat"] = acf_data[2]
                    group_data["pvalues"] = acf_data[3]
            else:
                if qstat:
                    group_data["qstat"] = acf_data[1]
                    group_data["pvalues"] = acf_data[2]
            group_acf_data[group] = group_data
        return group_acf_data

    @experimental
    def calculate_pacf(self, nlags=None, method="ywadjusted", alpha=None):
        """
        Utility for calculating the partial autocorrelation function for each group.
        In conjunction with the autocorrelation function ``calculate_acf``, the values returned
        from a pacf calculation can assist in setting values or bounds on AR, MA, and ARMA terms
        for an ARIMA model.

        The general rule to determine whether to use an AR, MA, or ARMA configuration for
        ``ARIMA`` (or ``AutoARIMA``) is as follows:

        * ACF gradually trend to significance, PACF significance achieved after 1 lag -> AR model
        * ACF significance after 1 lag, PACF gradually trend to significance -> MA model
        * ACF gradually trend to significance, PACF gradually trend to significance -> ARMA model

        These results can help to set the order terms of an ARIMA model (``p`` and ``q``) or,
        for ``AutoARIMA``, set restrictions on maximum search space terms to assist in faster
        optimization of the model.

        :param nlags: The count of partial autocorrelation lags to calculate and return.

                      Default: ``40``
        :param method: The method used for pacf calculation.
                       See the ``pmdarima`` docs for full listing of methods:

                       https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.pacf.html

                       Default: ``'ywadjusted'``
        :param alpha: If specified, returns confidence intervals based on the alpha value supplied.

                      Default: ``None``
        :return: Dictionary of ``{<group_key>: {<pacf terms>: <values as array>}}``
        """
        self._create_group_df()
        group_pacf_data = {}
        for group, df in self._group_df:
            pacf_data = pacf(x=df[self._y_col], nlags=nlags, method=method, alpha=alpha)
            group_data = (
                {"pacf": pacf_data[0]} if isinstance(pacf_data, tuple) else {"pacf": pacf_data}
            )
            if alpha:
                group_data["confidence_intervals"] = pacf_data[1]
            group_pacf_data[group] = group_data
        return group_pacf_data

    @experimental
    def generate_diff(self, lag=1, differences=1):
        """
        A utility for generating the array diff (lag differences) for each group.
        To support invertability, this method will return the starting value of each array as well
        as the differenced values.

        :param lag: Determines the magnitude of the lag to calculate the differencing function for.

                    Default: ``1``
        :param differences: The order of the differencing to be performed. Note that values > 1
                            will generate n fewer results.

                            Default: ``1``
        :return: Dictionary of ``{<group_key>: {"series_start": <float>, "diff": <diff_array>}}``
        """
        self._create_group_df()
        group_diff_data = {}
        for group, df in self._group_df:
            df.reset_index(inplace=True)
            group_data = {
                "diff": diff(x=df[self._y_col], lag=lag, differences=differences),
                "series_start": df[self._y_col][0],
            }
            group_diff_data[group] = group_data
        return group_diff_data

    @staticmethod
    @experimental
    def generate_diff_inversion(group_diff_data, lag=1, differences=1, recenter=False):
        """
        A utility for inverting a previously differenced group of timeseries data.
        This utility supports returning each group's series data to the original range of the data
        if the recenter argument is set to `True` and the start conditions are contained within
        the ``group_diff_data`` argument's dictionary structure.

        :param group_diff_data: Differenced payload consisting of a dictionary of
                                ``{<group_key>: {'diff': <differenced data>,
                                [optional]'series_start': float}}``
        :param lag: The lag to use to perform the differencing inversion.

                    Default: ``1``
        :param differences: The order of differencing to be used during the inversion.

                            Default: ``1``
        :param recenter: If ``True`` and ``'series_start'`` exists in ``group_diff_data`` dict,
                         will restore the original series range for each group based on the series
                         start value calculated through the ``generate_diff()`` method.
                         If the ``group_diff_data`` does not contain the starting values, the data
                         will not be re-centered.

                         Default: ``False``
        :return: Dictionary of ``{<group_key>: <series_inverted_data>}``
        """
        warn_check = False
        series_data = {}
        for group, payload in group_diff_data.items():
            data = payload.get("diff", None)
            if data is None:
                raise DivinerException(
                    f"group_diff_data does not contain the key `diff` for group" f"{group}"
                )
            inverted = diff_inv(x=data, lag=lag, differences=differences)
            if recenter:
                start = payload.get("series_start", None)
                if not start:
                    if not warn_check:
                        warnings.warn(
                            "Recentering is not possible due to `series_start` missing "
                            "from `group_diff_data` argument."
                        )
                        warn_check = True
                    series_data[group] = inverted
                else:
                    series_data[group] = inverted + start
            else:
                series_data[group] = inverted
        return series_data
