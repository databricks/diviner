.. _grouped_pmdarima:

Grouped pmdarima
================

The Grouped ``pmdarima`` API is a multi-series orchestration framework for building multiple individual models
of related, but isolated series data. For example, a project that required the forecasting of inventory demand at
regional warehouses around the world would historically require individual orchestration of data acquisition, hyperparameter
definitions, model training, metric validation, serialization, and registration of tens of thousands of individual
models based on the permutations of SKU and warehouse location.

This API consolidates the many thousands of models that would otherwise need to be implemented, trained individually,
and managed throughout their frequent retraining and forecasting lifecycles to a single high-level API that simplifies
these common use cases that rely on the `pmdarima <https://alkaline-ml.com/pmdarima/index.html>`_ forecasting library.

.. contents:: Table of Contents
    :local:
    :depth: 2

.. _grouped_pmdarima_api:

Grouped pmdarima API
--------------------
The following sections provide a basic overview of using the :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` API,
from fitting of the grouped models, predicting forecasted data, saving, loading, and customization of the underlying
``pmdarima`` instances.

To see working end-to-end examples, you can go to :ref:`tutorials-and-examples`. The examples will allow you
to explore the data structures required for training, how to extract forecasts for each group, and demonstrations of the
saving and loading of trained models.

Base Estimators and API interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The usage of the :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` API is slightly different from the other grouped
forecasting library wrappers within :py:mod:`Diviner <diviner>`. This is due to the ability of ``pmdarima`` to support
multiple modes of configuration.

These modes that are available to construct a model are:

* Passing an ``ARIMA`` model template (wrapper around `statsmodels ARIMA <https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html>`_)
* Using the native ``pmdarima`` ``AutoARIMA`` model `template <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA>`_
* Constructing a ``pmdarima`` `Pipeline template <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.pipeline.Pipeline.html#pmdarima.pipeline.Pipeline>`_

The :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` implementation requires the submission of one of these 3
model templates to set the base configured model architecture for each group.

For example:

.. code-block:: python

    from pmdarima.arima.arima import ARIMA
    from diviner import GroupedPmdarima

    # Define the base ARIMA with a preset ordering parameter
    base_arima_model = ARIMA(order=(1, 0, 2))

    # Define the model template in the GroupedPmdarima constructor
    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

The above example is intended only to showcase the interface between a base estimator (``base_arima_model``) and the
instance constructor for GroupedPmdarima. For a more in-depth and realistic example of utilizing an ARIMA model manually,
see the additional statistical validation steps that would be required for this in the :ref:`tutorials-and-examples` section
of the docs.

.. _pmdarima-fit:

Model fitting
^^^^^^^^^^^^^

In order to fit a :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` model instance, the :py:meth:`fit <diviner.GroupedPmdarima.fit>`
method is used. Calling this method will process the input ``DataFrame`` to create a grouped execution collection,
fit a ``pmdarima`` model type on each individual series, and persist the trained state of each group's model to the
object instance.

The arguments for the :py:meth:`fit <diviner.GroupedPmdarima.fit>` method are:

df
    A 'normalized' DataFrame that contains an endogenous regressor column (the 'y' column), a date (or datetime) column
    (that defines the ordering, periodicity, and frequency of each series (if this column is a string, the frequency will
    be inferred)), and grouping column(s) that define the discrete series to be modeled. For further information
    on the structure of this ``DataFrame``, see the :ref:`quickstart guide <quickstart>`

group_key_columns
    The names of the columns within ``df`` that, when combined (in order supplied) define distinct series. See the
    :ref:`quickstart guide <quickstart>` for further information.

y_col
    Name of the endogenous regressor term within the ``DataFrame`` argument ``df``. This column contains the values of
    the series that are used during training.

datetime_col
    Name of the column within the ``df`` argument ``DataFrame`` that defines the datetime ordering of the series data.

exog_cols
    *[Optional]* A collection of column names within the submitted data that contain exogenous regressor elements to
    use as part of model fitting and predicting. The data within each column will be assembled into a 2D array
    for use in the regression.

.. note::
    ``pmdarima`` currently has exogeneous regressor support marked as a future deprecated feature. Usage of this
    functionality is not recommended except for existing legacy implementations.


ndiffs
    *[Optional]* A dictionary of ``{<group_key>: <d value>}`` for the differencing term for each group.
    This is intended to function alongside the output from the :py:meth:`diviner.PmdarimaAnalyzer.calculate_ndiffs`
    method, serving to reduce the search space for ``AutoARIMA`` by supplying fixed ``d`` values to each group's model.

nsdiffs
    *[Optional]* A dictionary of ``{<group_key>: <D value>}`` for the seasonal differencing term for each group.
    This is intended to function alongside the output from the :py:meth:`diviner.PmdarimaAnalyzer.calculate_nsdiffs`
    method, serving to reduce the search space for ``AutoARIMA`` by supplying fixed ``D`` values to each group's model.

.. note::
    These values will only be used if the models being fit are seasonal models. The value ``m`` must be set on the
    underlying ``ARIMA`` or ``AutoARIMA`` model for seasonality order components to be used.

silence_warnings
    *[Optional]* Whether to silence stdout reporting of the underlying ``pmdarima`` fit process. Default: False.

fit_kwargs
    *[Optional]* ``fit_kwargs`` for ``pmdarima`` ``ARIMA``, ``AutoARIMA``, or ``Pipeline`` stages overrides.
    For more information, see the ``pmdarima`` `docs <https://alkaline-ml.com/pmdarima/index.html>`_

Example:

.. code-block:: python

    from pmdarima.arima.arima import ARIMA
    from diviner import GroupedPmdarima

    base_arima_model = ARIMA(order=(1, 0, 2))

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

.. _pmdarima-predict:

Predict
^^^^^^^

The :py:meth:`predict <diviner.GroupedPmdarima.predict>` method generates forecast data for each grouped series within
the meta :py:class:`diviner.GroupedPmdarima` model.

Example:

.. code-block:: python

    from pmdarima.arima.arima import ARIMA
    from diviner import GroupedPmdarima

    base_arima_model = ARIMA(order=(1, 0, 2))

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    forecasts = grouped_arima_model.predict(n_periods=30)

The arguments for the :py:meth:`predict <diviner.GroupedPmdarima.predict>` method are:

n_periods
    The number of future periods to generate from the end of each group's series. The first value of the prediction
    forecast series will begin at one periodicity value after the end of the training series.
    For example, if the training series was of daily data from 2019-10-01 to 2021-10-02, the start of the prediction
    series output would be 2021-10-03 and continue for ``n_periods`` days from that point.

predict_col
    *[Optional]* The name to use for the generated column containing forecasted data. Default: ``"yhat"``

alpha
    *[Optional]* Confidence interval significance value for error estimates. Default: ``0.05``.

.. note::
    ``alpha`` is only used if the boolean flag ``return_conf_int`` is set to ``True``.

return_conf_int
    *[Optional]* Boolean flag for whether or not to calculate confidence intervals for the predicted forecasts.
    If ``True``, the columns ``"yhat_upper"`` and ``"yhat_lower"`` will be added to the output ``DataFrame``
    for the upper and lower confidence intervals for the predictions.

inverse_transform
    *[Optional]* Used exclusively for ``Pipeline`` based models that include an endogeneous transformer such as
    ``BoxCoxEndogTransformer`` or ``LogEndogTransformer``. Default: ``True`` (although it only applies *if* the
    ``model_template`` type passed in is a ``Pipeline`` that contains a transformer).
    An inversion of the endogeneous regression term can be helpful for distributions that are highly non-normal.
    For further reading on what the purpose of these functions are, why they are used, and how they might be applicable
    to a given time series, see this `link <https://en.wikipedia.org/wiki/Data_transformation_(statistics)>`_.

exog
    *[Optional]* If the original model was trained with an exogeneous regressor elements, the prediction will require these
    2D arrays at prediction time. This argument is used to hold the 2D array of future exogeneous regressor values to be
    used in generating the prediction for the regressor.

predict_kwargs
    *[Optional]* Extra ``kwarg`` arguments for any of the transform stages of a ``Pipeline`` or for additional ``predict``
    ``kwargs`` to the model instance. ``Pipeline`` ``kwargs`` are specified in the manner of ``sklearn`` ``Pipeline``
    format (i.e., ``<stage_name>__<arg name>=<value>``. e.g., to change the values of a fourier transformer at prediction
    time, the override would be: ``{'fourier__n_periods': 45})``

Predict Groups
^^^^^^^^^^^^^^

The :py:meth:`predict_groups <diviner.GroupedPmdarima.predict_groups>` method generates forecast data for a subset of
groups that a :py:class:`diviner.GroupedPmdarima` model was trained upon.

Example:

.. code-block:: python

    from pmdarima.arima.arima import ARIMA
    from diviner import GroupedPmdarima

    base_model = ARIMA(order=(2, 1, 2))

    grouped_arima = GroupedPmdarima(model_template=base_model)

    model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    subset_forecasts = model.predict_groups(groups=[("US", "NY"), ("FR", "Paris"), ("UA", "Kyiv")], n_periods=90)

The arguments for the :py:meth:`predict_groups <diviner.GroupedPmdarima.predict_groups>` method are:

groups
    A collection of groups (or single group) to generate a forecast for. Structures available for input to this
    argument are: ``Tuple[str]`` or ``numpy.ndarray[str]`` for a single group; ``List[Tuple[str]]``, ``Set[Tuple[str]]``,
    or ``numpy.ndarray[numpy.ndarray[str]]`` for a collection of groups.

    .. note::
        Groups that are submitted for prediction that are not present in the trained model will, by default, cause an
        Exception to be raised. This behavior can be changed to a warning or ignore status with the argument ``on_error``.

n_periods
    The number of future periods to generate from the end of each group's series. The first value of the prediction
    forecast series will begin at one periodicity value after the end of the training series.
    For example, if the training series was of daily data from 2019-10-01 to 2021-10-02, the start of the prediction
    series output would be 2021-10-03 and continue for ``n_periods`` days from that point.

predict_col
    *[Optional]* The name to use for the generated column containing forecasted data. Default: ``"yhat"``

alpha
    *[Optional]* Confidence interval significance value for error estimates. Default: ``0.05``.

.. note::
    ``alpha`` is only used if the boolean flag ``return_conf_int`` is set to ``True``.

return_conf_int
    *[Optional]* Boolean flag for whether or not to calculate confidence intervals for the predicted forecasts.
    If ``True``, the columns ``"yhat_upper"`` and ``"yhat_lower"`` will be added to the output ``DataFrame``
    for the upper and lower confidence intervals for the predictions.

inverse_transform
    *[Optional]* Used exclusively for ``Pipeline`` based models that include an endogeneous transformer such as
    ``BoxCoxEndogTransformer`` or ``LogEndogTransformer``. Default: ``True`` (although it only applies *if* the
    ``model_template`` type passed in is a ``Pipeline`` that contains a transformer).
    An inversion of the endogeneous regression term can be helpful for distributions that are highly non-normal.
    For further reading on what the purpose of these functions are, why they are used, and how they might be applicable
    to a given time series, see this `link <https://en.wikipedia.org/wiki/Data_transformation_(statistics)>`_.

exog
    *[Optional]* If the original model was trained with an exogeneous regressor elements, the prediction will require these
    2D arrays at prediction time. This argument is used to hold the 2D array of future exogeneous regressor values to be
    used in generating the prediction for the regressor.

on_error
    *[Optional]* [Default -> ``"raise"``] Dictates the behavior for handling group keys that have been submitted in the
    ``groups`` argument that do not match with a group identified and registered during training (``fit``). The modes
    are:

    - ``"raise"``
        A :py:class:`DivinerException <diviner.exceptions.DivinerException>` is raised if any supplied
        groups do not match to the fitted groups.
    - ``"warn"``
        A warning is emitted (printed) and logged for any groups that do not match to those that the model
        was fit with.
    - ``"ignore"``
        Invalid groups will silently fail prediction.

    .. note::
        A :py:class:`DivinerException <diviner.exceptions.DivinerException>` will still be raised even in ``"ignore"``
        mode if there are no valid fit groups to match the provided ``groups`` provided to this method.

predict_kwargs
    *[Optional]* Extra ``kwarg`` arguments for any of the transform stages of a ``Pipeline`` or for additional ``predict``
    ``kwargs`` to the model instance. ``Pipeline`` ``kwargs`` are specified in the manner of ``sklearn`` ``Pipeline``
    format (i.e., ``<stage_name>__<arg name>=<value>``. e.g., to change the values of a fourier transformer at prediction
    time, the override would be: ``{'fourier__n_periods': 45})``

Save
^^^^
Saves a :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` instance that has been :py:meth:`fit <diviner.GroupedPmdarima.fit>`.
The serialization of the model instance uses a base64 encoding of the pickle serialization of each model instance within
the grouped structure.

Example:

.. code-block:: python

    from pmdarima.arima.arima import ARIMA
    from diviner import GroupedPmdarima

    base_arima_model = ARIMA(order=(1, 0, 2))

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    save_location = "/path/to/saved/model"

    grouped_arima_model.save(save_location)

Load
^^^^
Loads a :py:class:`GroupedPmdarima <diviner.GroupedPmdarima>` serialized model from a storage location.

Example:

.. code-block:: python

    from diviner import GroupedPmdarima

    load_location = "/path/to/saved/model"

    loaded_model = GroupedPmdarima.load(load_location)

.. note:: The :py:meth:`load <diviner.GroupedPmdarima.load>` method is a class method. As such, the initialization argument
    ``model_template`` does not need to be provided. It will be set on the loaded object based on the template that was
    provided during initial training before serialization.

Utilities
---------

Parameter Extraction
^^^^^^^^^^^^^^^^^^^^
To extract the parameters that are either explicitly (or, in the case of ``AutoARIMA``, selectively) set during the fitting
of each individual model contained within the grouped collection, the method :py:meth:`get_model_params <diviner.GroupedPmdarima.get_model_params>`
is used to extract the per-group parameters for each model into an output ``Pandas DataFrame``.

.. note:: The parameters can only be extracted from a ``GroupedPmdarima`` model that has been fit.

Example:

.. code-block:: python

    from pmdarima.arima.auto import AutoARIMA
    from diviner import GroupedPmdarima

    base_arima_model = AutoARIMA(max_order=7, d=1, m=7, max_iter=1000)

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    fit_parameters = grouped_arima_model.get_model_params()

Metrics Extraction
^^^^^^^^^^^^^^^^^^

This functionality allows for the retrieval of fitting metrics that are attached to the underlying ``SARIMA`` model.
These are not the typical loss metrics that can be calculated through cross validation backtesting.

The metrics that are returned from fitting are:

* `hqic <https://en.wikipedia.org/wiki/Hannan%E2%80%93Quinn_information_criterion>`_ (Hannan-Quinn information criterion)
* `aicc <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_ (Corrected Akaike information criterion; aic for small sample sizes)
* `oob <https://en.wikipedia.org/wiki/Out-of-bag_error>`_ (out of bag error)
* `bic <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ (Bayesian information criterion)
* `aic <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_ (Akaike information criterion)

.. note::
    Out of bag error metric (oob) is only calculated if the underlying ``ARIMA`` model has a value set for the argument
    ``out_of_sample_size``. See `link here <https://en.wikipedia.org/wiki/Out-of-bag_error>`_ for more information.

Example:

.. code-block:: python

    from pmdarima.arima.auto import AutoARIMA
    from diviner import GroupedPmdarima

    base_arima_model = AutoARIMA(max_order=7, d=1, m=7, max_iter=1000)

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    fit_metrics = grouped_arima_model.get_metrics()

Cross Validation Backtesting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cross validation utilizing `backtesting <https://en.wikipedia.org/wiki/Backtesting>`_ is the primary means of evaluating whether a given model will perform robustly in
generating forecasts based on time period horizon events throughout the historical series.

In order to use the cross validation functionality in the method :py:meth:`diviner.GroupedPmdarima.cross_validate`,
one of two windowing split objects must be passed into the method signature:

* `RollingForecastCV <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.RollingForecastCV.html#pmdarima.model_selection.RollingForecastCV>`_
* `SlidingWindowForecastCV <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.SlidingWindowForecastCV.html#pmdarima.model_selection.SlidingWindowForecastCV>`_

Arguments to the :py:meth:`diviner.GroupedPmdarima.cross_validate` method:

df
    The original source ``DataFrame`` that was used during :py:meth:`diviner.GroupedPmdarima.fit` that contains
    the endogenous series data. This ``DataFrame`` must contain the columns that define the constructed groups
    (i.e., missing group data will not be scored and groups that are not present in the model object will raise an Exception).

metrics
    A collection of metric names to be used for evaluation. Submitted metrics must be one or more of:

    * `smape <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_
    * `mean_absolute_error <https://en.wikipedia.org/wiki/Mean_absolute_error>`_
    * `mean_squared_error <https://en.wikipedia.org/wiki/Mean_squared_error>`_

    Default: ``{"smape", "mean_absolute_error", "mean_squared_error"}``

cross_validator
    The cross validation object (either ``RollingForecastCV`` or ``SlidingWindowForecastCV``). See the example below
    for how to submit this object to the :py:meth:`diviner.GroupedPmdarima.cross_validate` method.

error_score
    The default value to assign to a window evaluation if an error occurs during loss calculation.

    Default: ``np.nan``

    In order to throw an Exception, a str value of "raise" can be provided. Otherwise, supply a float.

verbosity
    Level of verbosity to print during training and cross validation. The lower the integer value, the fewer
    lines of debugging text is printed to stdout.

    Default: ``0`` (no printing)

Example:

.. code-block:: python

    from pmdarima.arima.auto import AutoARIMA
    from pmdarima.model_selection import SlidingWindowForecastCV
    from diviner import GroupedPmdarima

    base_arima_model = AutoARIMA(max_order=7, d=1, m=7, max_iter=1000)

    grouped_arima = GroupedPmdarima(model_template=base_arima_model)

    grouped_arima_model = grouped_arima.fit(df, ["country", "region"], "sales", "date")

    cv_window = SlidingWindowForecastCV(h=28, step=180, window_size=365)

    grouped_arima_cv = grouped_arima_model.cross_validate(df=df,
                                                          metrics=["mean_squared_error", "smape"],
                                                          cross_validator=cv_window,
                                                          error_score=np.nan,
                                                          verbosity=1
                                                         )

Class Signature of GroupedPmdarima
----------------------------------

.. autoclass:: diviner.GroupedPmdarima
    :members:


Grouped pmdarima Analysis tools
-------------------------------

.. warning::
    The ``PmdarimaAnalyzer`` module is in experimental mode. The methods and signatures are subject to change in the
    future with no deprecation warnings.

As a companion to ``Diviner``'s :py:class:`diviner.GroupedPmdarima` class, an analysis toolkit class is provided.
Contained within this class, :py:class:`PmdarimaAnalyzer <diviner.PmdarimaAnalyzer>`, are the following utility methods:

* `decompose_groups <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.decompose.html#pmdarima.arima.decompose>`_
* `calculate_ndiffs <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html#pmdarima.arima.ndiffs>`_
* `calculate_nsdiffs <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.nsdiffs.html#pmdarima.arima.nsdiffs>`_
* `calculate_is_constant <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.is_constant.html#pmdarima.arima.is_constant>`_
* `calculate_acf <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.acf.html#pmdarima.utils.acf>`_
* `calculate_pacf <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.plot_pacf.html#pmdarima.utils.plot_pacf>`_

See below for a brief description of each of these utility methods that are available for group processing through the
``PmdarimaAnalyzer`` API.

Object instantiation:

.. code-block:: python

    from diviner import PmdarimaAnalyzer

    analyzer = PmdarimaAnalyzer(
        df=df,
        group_key_columns=["country", "region"],
        y_col="orders",
        datetime_col="date"
    )

Decompose Trends
^^^^^^^^^^^^^^^^

The :py:meth:`diviner.PmdarimaAnalyzer.decompose_groups` method will decompose each series into its component parts:

* trend
* seasonal
* random (also known as 'residuals')

The output of this method is a union of each group's decomposed trends in a single ``DataFrame`` that retains the group
key information in columns along with the extracted components from the series data.

This method is mainly used for validation of a new project.

Example:

.. code-block:: python

    decomposed_trends = analyzer.decompose_groups(m=7, type="additive")

Arguments to the :py:meth:`diviner.PmdarimaAnalyzer.decompose_groups` method:

m
    The frequency value of the endogenous series data. The integer supplied is a measure of the repeatable pattern of
    the estimated seasonality effect. For instance, ``7`` would be appropriate for daily measured data, ``24`` would be
    a good starting point for hourly data, and ``52`` would be a good initial validation value for weekly data.

type (``'type_'``)
    The type of decomposition to perform. One of: ``"additive"`` or ``"multiplicative"``.
    A good rule of thumb for determining which of these to choose is to determine whether the seasonality effects
    either stay constant as a function of the trend (which would be "additive") or, if the seasonality effect is a function
    of the baseline trend value, "multiplicative" would be more appropriate.
    For further explanation, see `here <https://anomaly.io/seasonal-trend-decomposition-in-r/index.html>`_.

filter (``'filter_'``)
    *[Optional]* Reverse-sorted Array for performing convolution on the coefficients of either the ``MA`` terms or the ``AR`` terms.

    Default: ``None``

Calculate Differencing Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isolating the differencing term ``'d'`` can provide significant performance improvements if ``AutoARIMA`` is used as the
underlying estimator for each series. This method provides a means of estimating these per-group differencing terms.

The output is returned as a dictionary of ``{<group_key>: d}``

.. note::
    This utility method is intended to be used as an input to the :py:meth:`diviner.GroupedPmdarima.fit` method when using
    ``AutoARIMA`` as a base group estimator. It will set *per-group* values of ``d`` so that the AutoARIMA optimizer
    does not need to search for values of the differencing term, saving a great deal of computation time.

Example:

.. code-block:: python

    diffs = analyzer.calculate_ndiffs(alpha=0.1, test="kpss", max_d=5)

Arguments to the :py:meth:`diviner.PmdarimaAnalyzer.calculate_ndiffs` method:

alpha
    The significance value used in determining if a pvalue for a test of an estimated ``d`` term is significant or not.
    Default: ``0.05``

test
    The stationarity unit test used to determine significance for a tested ``d`` term.

    Allowable values:

    * `"kpss" <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.html#pmdarima.arima.KPSSTest>`_
    * `"pp" <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html#pmdarima.arima.PPTest>`_
    * `"adf" <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.html#pmdarima.arima.ADFTest>`_

    Default: ``"kpss"``

max_d
    The maximum allowable differencing term to test.
    Default: ``2``

Calculate Seasonal Differencing Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isolating the seasonal differencing term ``D`` can provide a significant performance improvement to seasonal models
which are activated by setting the ``m`` term in the base group estimator. The functionality of this
:py:meth:`diviner.PmdarimaAnalyzer.calculate_nsdiffs` method is similar to that of ``calculate_ndiffs``, except for the
seasonal differencing term.

Example:

.. code-block:: python

    seasonal_diffs = analyzer.calculate_nsdiffs(m=7, test="ocsb", max_D=5)

Arguments to the ``calculate_nsdiffs`` method:

m
    The frequency of seasonal periods within the endogenous series. The integer supplied is a measure of the repeatable pattern of
    the estimated seasonality effect. For instance, ``7`` would be appropriate for daily measured data, ``24`` would be
    a good starting point for hourly data, and ``52`` would be a good initial validation value for weekly data.

test
    The seasonality unit test used to determine an optimal seasonal differencing ``D`` term.

    Allowable tests:

    * `"ocsb" <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html#pmdarima.arima.OCSBTest>`_
    * `"ch" <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html#pmdarima.arima.CHTest>`_

    Default: ``"ocsb"``

max_D
    The maximum allowable seasonal differencing term to test.

    Default: ``2``

Calculate Constancy
^^^^^^^^^^^^^^^^^^^

The constancy check is a data set utility validation tool that operates on each grouped series,
determining whether or not it can be modeled.

The output of this validation check method :py:meth:`diviner.PmdarimaAnalyzer.calculate_is_constant` is a dictionary of
``{<group_key>: <Boolean constancy check>}``. Any group with a ``True`` result **is ineligible for modeling** as this
indicates that the group has only a single constant value for each datetime period.

Example:

.. code-block:: python

    constancy_checks = analyzer.calculate_is_constant()


.. _pmdarima_acf:

Calculate Auto Correlation Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`diviner.PmdarimaAnalyzer.calculate_acf` method is used for calculating the auto-correlation function for
each series group. The auto-correlation function values can be used (in conjunction with the result of partial
auto-correlation function results) to select restrictive search values for the ordering
terms for ``AutoARIMA`` or to manually set the ordering terms (``(p, d, q)``) for ``ARIMA``.

.. note::
    The general rule to determine whether to use an AR, MA, or ARMA configuration for ``ARIMA`` or ``AutoARIMA`` is
    as follows:

    * ACF gradually trend to significance, PACF significance achieved after 1 lag -> AR model
    * ACF significance after 1 lag, PACF gradually trend to significance -> MA model
    * ACF gradually trend to significance, PACF gradually trend to significance -> ARMA model

    These results can help to set the order terms of an ARIMA model (p and q) or,
    for AutoARIMA, set restrictions on maximum search space terms to assist in faster
    optimization of the model.

Arguments to the ``calculate_acf`` method:

unbiased
    auto-covariance denominator flag with values of:

    * ``True`` -> denominator = ``n - k``
    * ``False`` -> denominator = ``n``

nlags
    The number of auto-correlation lags to calculate and return.

    Default: ``40``

qstat
    Boolean flag to calculate and return the Q statistic from the `Ljung-Box test <https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test>`_.

    Default: ``False``

fft
    Whether to perform a fast fourier transformation of the series to calculate the auto-correlation function. For large
    time series, it is highly recommended to set this to ``True``. Allowable values: ``True``, ``False``, or ``None``.

    Default: ``None``

alpha
    If specified as a float, calculates and returns confidence intervals at this certainty level for the auto-correlation function
    values. For example, if alpha=0.1, 90% confidence intervals are calculated and returned wherein the standard deviation is computed
    according to `Bartlett's formula <https://en.wikipedia.org/wiki/Bartlett%27s_test>`_.

    Default: ``None``

missing
    Handling of ``NaN`` values in series data. Available options are:

    * ``None`` - no validation checks are performed.
    * ``'raise'`` - an Exception is raised if a missing value is detected.
    * ``'conservative'`` - ``NaN`` values are removed from the mean and cross-product calculations but are not removed from the series data.
    * ``'drop'`` - ``NaN`` values are removed from the series data.

    Default: ``None``

adjusted
    Deprecation handler for the underlying ``statsmodels`` arguments that have become the ``unbiased`` argument. This
    is a duplicated value for the denominator mode of calculation for the autocovariance of the series.

    Default: ``False``

Calculate Partial Auto Correlation Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`diviner.PmdarimaAnalyzer.calculate_pacf` method is used for determining the partial auto-correlation function
for each series group. When combined with :ref:`pmdarima_acf` results, ordering values can be estimated (or controlled
in search space scope for ``AutoARIMA``). See the notes in :ref:`pmdarima_acf` for how to use the results from these two methods.

Arguments to the ``calculate_pacf`` method:

nlags
    The number of partial auto-correlation lags to calculate and return.

    Default: ``40``

method
    The method employed for calculating the partial auto-correlation function.
    Methods and their explanations are listed in `the pmdarima docs <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.utils.pacf.html>`_.

    Default: ``'ywadjusted'``

alpha
    If specified as a float, calculates and returns confidence intervals at this certainty level for the auto-correlation function
    values. For example, if alpha=0.1, 90% confidence intervals are calculated and returned wherein the standard deviation is computed
    according to `Bartlett's formula <https://en.wikipedia.org/wiki/Bartlett%27s_test>`_.

    Default: ``None``

Generate Diff
^^^^^^^^^^^^^

The utility method :py:meth:`diviner.PmdarimaAnalyzer.generate_diff` will generate lag differences for each group.
While not applicable to most timeseries modeling problems, it can prove to be useful in certain situations or as a
diagnostic tool to investigate why a particular series is not fitting properly.

Arguments for this method:

lag
    The magnitude of the lag used in calculating the differencing.
    Default: ``1``

differences
    The order of the differencing to be performed.
    Default: ``1``

For an illustrative example, see `the diff example <https://alkaline-ml.com/pmdarima/auto_examples/utils/example_array_differencing.html#sphx-glr-auto-examples-utils-example-array-differencing-py>`_.

Generate Diff Inversion
^^^^^^^^^^^^^^^^^^^^^^^

The utility method :py:meth:`diviner.PmdarimaAnalyzer.generate_diff_inversion` will invert a previously differenced
grouped series.

Arguments for this method:

group_diff_data
    The differenced data from the usage of :py:meth:`diviner.PmdarimaAnalyzer.generate_diff`.

lag
    The magnitude of the lag that was used in the differencing function in order to revert the diff.

    Default: ``1``

differences
    The order of the differencing that was performed using :py:meth:`diviner.PmdarimaAnalyzer.generate_diff` so that the
    series data can be reverted.

    Default: ``1``

recenter
    If ``True`` and ``'series_start'`` exists in ``group_diff_data`` dict, will restore the original series range for
    each group based on the series start value calculated through the ``generate_diff()`` method.
    If the ``group_diff_data`` does not contain the starting values, the data will not be re-centered.

    Default: ``False``

Class Signature of PmdarimaAnalyzer
-----------------------------------

.. autoclass:: diviner.PmdarimaAnalyzer
    :members: