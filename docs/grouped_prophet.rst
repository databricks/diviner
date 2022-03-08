.. _grouped_prophet:

Grouped Prophet
===============

The Grouped Prophet model is a multi-series orchestration framework for building multiple individual models
of related, but isolated series data. For example, a project that required the forecasting of airline passengers at
major airports around the world would historically require individual orchestration of data acquisition, hyperparameter
definitions, model training, metric validation, serialization, and registration of thousands of individual models.

This API consolidates the many thousands of models that would otherwise need to be implemented, trained individually,
and managed throughout their frequent retraining and forecasting lifecycles to a single high-level API that simplifies
these common use cases that rely on the `Prophet <https://facebook.github.io/prophet/>`_ forecasting library.

.. contents:: Table of Contents
    :local:
    :depth: 2

.. _api:

Grouped Prophet API
-------------------
The following sections provide a basic overview of using the :py:class:`GroupedProphet <diviner.GroupedProphet>` API,
from fitting of the grouped models, predicting forecasted data, saving, loading, and customization of the underlying
``Prophet`` instances.

To see working end-to-end examples, you can go to :ref:`tutorials-and-examples`. The examples will allow you
to explore the data structures required for training, how to extract forecasts for each group, and demonstrations of the
saving and loading of trained models.

.. _fit:

Model fitting
^^^^^^^^^^^^^

In order to fit a :py:class:`GroupedProphet <diviner.GroupedProphet>` model instance, the :py:meth:`fit <diviner.GroupedProphet.fit>`
method is used. Calling this method will process the input ``DataFrame`` to create a grouped execution collection,
fit a ``Prophet`` model on each individual series, and persist the trained state of each group's model to the
object instance.

The arguments for the :py:meth:`fit <diviner.GroupedProphet.fit>` method are:

df
    A 'normalized' DataFrame that contains an endogenous regressor column (the 'y' column), a date (or datetime) column
    (that defines the ordering, periodicity, and frequency of each series (if this column is a string, the frequency will
    be inferred)), and grouping column(s) that define the discrete series to be modeled. For further information
    on the structure of this ``DataFrame``, see the :ref:`quickstart guide <quickstart>`

group_key_columns
    The names of the columns within ``df`` that, when combined (in order supplied) define distinct series. See the
    :ref:`quickstart guide <quickstart>` for further information.

kwargs
    *[Optional]* Arguments that are used for overrides to the ``Prophet`` pystan optimizer. Details of what parameters are available
    and how they might affect the optimization of the model can be found by running
    ``help(pystan.StanModel.optimizing)`` from a Python REPL.

Example:

.. code-block:: python

    grouped_prophet_model = GroupedProphet().fit(df, ["country", "region"])

.. _forecast:

Forecast
^^^^^^^^
The :py:meth:`forecast <diviner.GroupedProphet.forecast>` method is the 'primary means' of generating future forecast
predictions. For each group that was trained in the :ref:`fit` of the grouped model,
a value of time periods is predicted based upon the last event date (or datetime) from each series' temporal
termination.

Usage of this method requires providing two arguments:

horizon
    The number of events to forecast (supplied as a positive integer)
frequency
    The periodicity between each forecast event. Note that this value does not have to match the periodicity of the
    training data (i.e., training data can be in days and predictions can be in months, minutes, hours, or years).

    The frequency abbreviations that are allowed can be found
    `here. <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_

.. note:: The generation of error estimates (`yhat_lower` and `yhat_upper`) in the output of a forecast are controlled
    through the use of the ``Prophet`` argument ``uncertainty_samples`` during class instantiation, prior to :ref:`fit`
    being called. Setting this value to `0` will eliminate error estimates and will dramatically increase the speed of
    training, prediction, and cross validation.

The return data structure for this method will be of a 'stacked' ``pandas`` ``DataFrame``, consisting of the
grouping keys defined (in the order in which they were generated), the grouping columns, elements of the prediction
values (deconstructed; e.g. 'weekly', 'yearly', 'daily' seasonality terms and the 'trend'), the date (datetime) values,
and the prediction itself (labeled `yhat`).

.. _predict:

Predict
^^^^^^^
A 'manual' method of generating predictions based on discrete date (or datetime) values for each group specified.
This method accepts a ``DataFrame`` as input having columns that define discrete dates to generate predictions for
and the grouping key columns that match those supplied when the model was fit.
For example, a model trained with the grouping key columns of 'city' and 'country' that included New York City, US
and Toronto, Canada as series would generate predictions for both of these cities if the provided
``df`` argument were supplied:

.. code-block:: python

    predict_config = pd.DataFrame.from_records(
        {
            "country": ["us", "us", "ca", "ca"],
            "city": ["nyc", "nyc", "toronto", "toronto"],
            "ds": ["2022-01-01", "2022-01-08", "2022-01-01", "2022-01-08"],
        }
    )

    grouped_prophet_model.predict(predict_config)

The structure of this submitted ``DataFrame`` for the above use case is:

.. list-table:: Predict `df` Structure
    :widths: 25 25 40
    :header-rows: 1

    * - country
      - city
      - ds
    * - us
      - nyc
      - 2022-01-01
    * - us
      - nyc
      - 2022-01-08
    * - ca
      - toronto
      - 2022-01-01
    * - ca
      - toronto
      - 2022-01-08

Usage of this method with the above specified df would generate 4 individual predictions; one for each row.

.. note:: The :ref:`forecast` method is more appropriate for most use cases as it will continue immediately after the
    training period of data terminates.

Predict Groups
^^^^^^^^^^^^^^

The :py:meth:`predict_groups <diviner.GroupedProphet.predict_groups>` method generates forecast data for a subset of
groups that a :py:class:`diviner.GroupedProphet` model was trained upon.

Example:

.. code-block:: python

    from diviner import GroupedProphet

    model = GroupedProphet().fit(df, ["country", "region"])

    subset_forecasts = model.predict_groups(groups=[("US", "NY"), ("FR", "Paris"), ("UA", "Kyiv")],
                                            horizon=90,
                                            frequency="D",
                                            on_error="warn"
                                            )

The arguments for the :py:meth:`predict_groups <diviner.GroupedProphet.predict_groups>` method are:

groups
    A collection of groups (or single group) to generate a forecast for. Structures available for input to this
    argument are: ``Tuple[str]`` or ``numpy.ndarray[str]`` for a single group; ``List[Tuple[str]]``, ``Set[Tuple[str]]``,
    or ``numpy.ndarray[numpy.ndarray[str]]`` for a collection of groups.

    .. note::
        Groups that are submitted for prediction that are not present in the trained model will, by default, cause an
        Exception to be raised. This behavior can be changed to a warning or ignore status with the argument ``on_error``.

horizon
    The number of events to forecast (supplied as a positive integer)

frequency
    The periodicity between each forecast event. Note that this value does not have to match the periodicity of the
    training data (i.e., training data can be in days and predictions can be in months, minutes, hours, or years).

    The frequency abbreviations that are allowed can be found
    `here. <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_

predict_col
    *[Optional]* The name to use for the generated column containing forecasted data. Default: ``"yhat"``

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

Save
^^^^
Supports saving a :py:class:`GroupedProphet <diviner.GroupedProphet>` model that has been :py:meth:`fit <diviner.GroupedProphet.fit>`.
The serialization of the model instance does not rely on pickle or cloudpickle, rather a straight-forward json
serialization.

.. code-block:: python

    save_location = "/path/to/store/model"
    grouped_prophet_model.save(save_location)

Load
^^^^
Loading a saved :py:class:`GroupedProphet <diviner.GroupedProphet>` model is done through the use of a class method. The
:py:meth:`load <diviner.GroupedProphet.load>` method is called as below:

.. code-block:: python

    load_location = "/path/to/stored/model"
    grouped_prophet_model = GroupedProphet.load(load_location)

.. note:: The ``PyStan`` backend optimizer instance used to fit the model is not saved (this would require compilation of
    ``PyStan`` on the same machine configuration that was used to fit it in order for it to be valid to reuse) as it is
    not useful to store and would require additional dependencies that are not involved in cross validation, parameter
    extraction, forecasting, or predicting. If you need access to the ``PyStan`` backend, retrain the model and access
    the underlying solver prior to serializing to disk.

Overriding Prophet settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create a :py:class:`GroupedProphet <diviner.GroupedProphet>` instance, there are no required attributes to
define. Utilizing the default values will, as with the underlying ``Prophet`` library, utilize the default values to
perform model fitting.
However, there are arguments that can be overridden which are pass-through values to the individual ``Prophet``
instances that are created for each group. Since these are ``**kwargs`` entries, the names will be argument names for
the respective arguments in ``Prophet``.

To see a full listing of available arguments for the given version of ``Prophet`` that you are using, the simplest
(as well as the recommended manner in the library documentation) is to run a ``help()`` command in a Python REPL:

.. code-block:: python

    from prophet import Prophet
    help(Prophet)

An example of overriding many of the arguments within the underlying ``Prophet`` model for the ``GroupedProphet`` API
is shown below.

.. code-block:: python

    grouped_prophet_model = GroupedProphet(
        growth='linear',
        changepoints=None,
        n_changepoints=90,
        changepoint_range=0.8,
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        holidays=None,
        seasonality_mode='additive',
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=None
    )

Utilities
---------

Parameter Extraction
^^^^^^^^^^^^^^^^^^^^
The method :py:meth:`extract_model_params <diviner.GroupedProphet.extract_model_params>` is a utility that extracts the tuning parameters
from each individual model from within the :ref:`model's <api>` container and returns them as a single DataFrame.
Columns are the parameters from the models, while each row is an individual group's Prophet model's parameter values.
Having a single consolidated extraction data structure eases the historical registration of model performance and
enables a simpler approach to the design of frequent retraining through passive retraining systems (allowing for
an easier means by which to acquire priors hyperparameter values on frequently retrained forecasting models).

An example extract from a 2-group model (cast to a dictionary from the ``Pandas DataFrame`` output) is shown below:

.. code-block:: python

    {'changepoint_prior_scale': {0: 0.05, 1: 0.05},
     'changepoint_range': {0: 0.8, 1: 0.8},
     'component_modes': {0: {'additive': ['yearly',
                                          'weekly',
                                          'additive_terms',
                                          'extra_regressors_additive',
                                          'holidays'],
                             'multiplicative': ['multiplicative_terms',
                                                'extra_regressors_multiplicative']},
                         1: {'additive': ['yearly',
                                          'weekly',
                                          'additive_terms',
                                          'extra_regressors_additive',
                                          'holidays'],
                             'multiplicative': ['multiplicative_terms',
                                                'extra_regressors_multiplicative']}},
     'country_holidays': {0: None, 1: None},
     'daily_seasonality': {0: 'auto', 1: 'auto'},
     'extra_regressors': {0: OrderedDict(), 1: OrderedDict()},
     'fit_kwargs': {0: {}, 1: {}},
     'grouping_key_columns': {0: ('key2', 'key1', 'key0'),
                              1: ('key2', 'key1', 'key0')},
     'growth': {0: 'linear', 1: 'linear'},
     'holidays': {0: None, 1: None},
     'holidays_prior_scale': {0: 10.0, 1: 10.0},
     'interval_width': {0: 0.8, 1: 0.8},
     'key0': {0: 'T', 1: 'M'},
     'key1': {0: 'A', 1: 'B'},
     'key2': {0: 'C', 1: 'L'},
     'logistic_floor': {0: False, 1: False},
     'mcmc_samples': {0: 0, 1: 0},
     'n_changepoints': {0: 90, 1: 90},
     'seasonality_mode': {0: 'additive', 1: 'additive'},
     'seasonality_prior_scale': {0: 10.0, 1: 10.0},
     'specified_changepoints': {0: False, 1: False},
     'stan_backend': {0: <prophet.models.PyStanBackend object at 0x7f900056d2e0>,
                      1: <prophet.models.PyStanBackend object at 0x7f9000523eb0>},
     'start': {0: Timestamp('2018-01-02 00:02:00'),
               1: Timestamp('2018-01-02 00:02:00')},
     't_scale': {0: Timedelta('1459 days 00:00:00'),
                 1: Timedelta('1459 days 00:00:00')},
     'train_holiday_names': {0: None, 1: None},
     'uncertainty_samples': {0: 1000, 1: 1000},
     'weekly_seasonality': {0: 'auto', 1: 'auto'},
     'y_scale': {0: 1099.9530489951537, 1: 764.727400507604},
     'yearly_seasonality': {0: 'auto', 1: 'auto'}}

.. _cv_score:

Cross Validation and Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The primary method of evaluating model performance across all groups is by using the method
:py:meth:`cross_validate_and_score <diviner.GroupedProphet.cross_validate_and_score>`. Using this method from a ``GroupedProphet`` instance
that has been fit will perform backtesting of each group's model using the training data set supplied when the
:py:meth:`fit <diviner.GroupedProphet.fit>` method was called.


The return type of this method is a single consolidated ``Pandas DataFrame`` that contains metrics as columns with
each row representing a distinct grouping key.
For example, below is a sample of 3 groups' cross validation metrics.

.. code-block:: python

    {'coverage': {0: 0.21839080459770113,
              1: 0.057471264367816084,
              2: 0.5114942528735632},
     'grouping_key_columns': {0: ('key2', 'key1', 'key0'),
                              1: ('key2', 'key1', 'key0'),
                              2: ('key2', 'key1', 'key0')},
     'key0': {0: 'T', 1: 'M', 2: 'K'},
     'key1': {0: 'A', 1: 'B', 2: 'S'},
     'key2': {0: 'C', 1: 'L', 2: 'Q'},
     'mae': {0: 14.230668998203283, 1: 34.62100210053155, 2: 46.17014668092673},
     'mape': {0: 0.015166533573997266,
              1: 0.05578282899646585,
              2: 0.047658812366283436},
     'mdape': {0: 0.013636314354422746,
               1: 0.05644041426067295,
               2: 0.039153745874603914},
     'mse': {0: 285.42142900120183, 1: 1459.7746527190932, 2: 3523.9281809854906},
     'rmse': {0: 15.197908800171147, 1: 35.520537302480314, 2: 55.06313841955681},
     'smape': {0: 0.015327226830099487,
               1: 0.05774645767583018,
               2: 0.0494437278595581}}

Method arguments:

horizon
    A ``pandas.Timedelta`` string consisting of two parts: an integer and a periodicity. For example, if the training
    data is daily, consists of 5 years of data, and the end-use for the project is to predict 14 days of future values
    every week, a plausible horizon value might be ``"21 days"`` or ``"28 days"``.
    See `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_ for information on
    the allowable syntax and format for ``pandas.Timedelta`` values.

metrics
    A list of metrics that will be calculated following the back-testing cross validation. By default, all of the
    following will be tested:

* "mae" (`mean absolute error <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error>`_)
* "mape" (`mean absolute percentage error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error>`_)
* "mdape" (median absolute percentage error)
* "mse" (`mean squared error <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error>`_)
* "rmse" (`root mean squared error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_)
* "smape" (`symmetric mean absolute percentage error <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_)

To restrict the metrics computed and returned, a subset of these tests can be supplied to the ``metrics`` argument.

period
    The frequency at which each windowed collection of back testing cross validation will be conducted. If the argument
    ``cutoffs`` is left as ``None``, this argument will determine the spacing between training and validation sets
    as the cross validation algorithm steps through each series. Smaller values will increase cross validation
    execution time.

initial
    The size of the initial training period to use for cross validation windows. The default derived value, if not
    specified, is ``horizon`` * 3 with cutoff values for each window set at ``horizon`` / 2.

parallel
    Mode of operation for calculating cross validation windows. ``None`` for serial execution, ``'processes'`` for
    multiprocessing pool execution, and ``'threads'`` for thread pool execution.

cutoffs
    Optional control mode that allows for defining specific datetime values in ``pandas.Timestamp`` format to determine
    where to conduct train and test split boundaries for validation of each window.

kwargs
    Individual optional overrides to ``prophet.diagnostics.cross_validation()`` and
    ``prophet.diagnostics.performance_metrics()`` functions. See the
    `prophet docs <https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation>`_ for more information.

.. _cv:

Cross Validation
^^^^^^^^^^^^^^^^
The :py:meth:`diviner.GroupedProphet.cross_validate` method is a wrapper around the ``Prophet`` function
``prophet.diagnostics.cross_validation()``. It is intended to be used as a debugging tool for the 'automated' metric
calculation method, see :ref:`Cross Validation and Scoring <cv_score>`. The arguments for this
method are:

horizon
    A timedelta formatted string in the ``Pandas.Timedelta`` format that defines the amount of time to utilize
    for generating a validation dataset that is used for calculating loss metrics per each cross validation window
    iteration. Example horizons: (``"30 days"``, ``"24 hours"``, ``"16 weeks"``). See
    `the pandas Timedelta docs <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_ for more
    information on supported formats and syntax.

period
    The periodicity of how often a windowed validation will be constructed. Smaller values here will take longer as
    more 'slices' of the data will be made to calculate error metrics. The format is the same as that of the horizon
    (i.e. ``"60 days"``).

initial
    The minimum size of data that will be used to build the cross validation window. Values that are excessively small
    may cause issues with the effectiveness of the estimated overall prediction error and lead to long cross validation
    runtimes. This argument is in the same format as ``horizon`` and ``period``, a ``pandas.Timedelta`` format string.

parallel
    Selection on how to execute the cross validation windows. Supported modes: (``None``, ``'processes'``, or
    ``'threads'``). Due to the reuse of the originating dataset for window slice selection, a shared memory instance
    mode ``'threads'`` is recommended over using ``'processes'`` mode.

cutoffs
    Optional arguments for specified ``pandas.Timestamp`` values to define where boundaries should be within
    the group series values. If this is specified, the ``period`` and ``initial`` arguments are not used.

.. note:: For information on how cross validation works within the ``Prophet`` library, see this
    `link <https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation>`_.

The return type of this method is a dictionary of ``{<group_key>: <pandas DataFrame>}``, the ``DataFrame`` containing
the cross validation window scores across time horizon splits.

Performance Metrics
^^^^^^^^^^^^^^^^^^^
The :py:meth:`calculate_performance_metrics <diviner.GroupedProphet.calculate_performance_metrics>` method is a
debugging tool that wraps the function `performance_metrics <https://facebook.github.io/prophet/docs/diagnostics.html>`_
from ``Prophet``. Usage of this method will generate the defined metric scores for each cross validation window,
returning a dictionary of ``{<group_key>: <DataFrame of metrics for each window>}``

Method arguments:

cv_results
    The output of :py:meth:`cross_validate <diviner.GroupedProphet.cross_validate>`.

metrics
    Optional subset list of metrics. See the signature for :ref:`cross_validate_and_score() <cv_score>` for supported
    metrics.

rolling_window
    Defines the fractional amount of data to use in each rolling window to calculate the performance metrics.
    Must be in the range of {0: 1}.

monthly
    Boolean value that, if set to ``True``, will collate the windows to ensure that horizons are computed as a factor
    of months of the year from the cutoff date. This is only useful if the data has a yearly seasonality component to it
    that relates to day of month.

Class Signature
---------------

.. autoclass:: diviner.GroupedProphet
    :members: