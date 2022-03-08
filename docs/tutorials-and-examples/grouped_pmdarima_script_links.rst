GroupedPmdarima Example Scripts
===============================

The scripts included below show the various options available for utilizing the ``GroupedPmdarima`` API.
For an alternative view of these examples with data visualizations, see the :ref:`notebooks here <tutorials-and-examples>`

.. contents:: Scripts
    :local:
    :depth: 2

.. _arima_script:

GroupedPmdarima ARIMA
---------------------

This example shows using a manually-configured (order values provided for a non-seasonal collection of series) ARIMA
model that is applied to each group.

Using this approach (a static order configuration) can be useful for homogenous collections of series. If each member of
the grouped collection of series shares a common characteristic in the residuals (i.e., the differencing terms for both
an auto-correlation and partial auto-correlation analysis shows similar relationships for all groups), this approach
will be faster and less expensive to fit a model than any other means.

.. literalinclude:: /../examples/grouped_pmdarima/arima_example.py
    :caption: GroupedPmdarima manually configured ARIMA model
    :language: python
    :linenos:

GroupedPmdarima AutoARIMA
-------------------------

For projects that do not have homogeneous relationships amongst groups of series, using the AutoARIMA functionality
of pmdarima is advised. This will allow for individualized optimation of the order terms (p, d, q) and, for seasonal
series, the (P, D, Q) seasonal order terms as well.

.. note::
    If using a seasonal approach, the parameter ``m`` must be set to an integer value that represents the seasonal
    periodicity. In this mode, with ``m`` set, the ARIMA terms (p, d, q) will be optimized along with (P, D, Q). Due to
    the complexity of optimizing these terms, this execution mode will take far longer than an optimization of a
    non-seasonal model.

.. literalinclude:: /../examples/grouped_pmdarima/autoarima_example.py
    :caption: GroupedPmdarima non-seasonal AutoARIMA model
    :language: python
    :linenos:

GroupedPmdarima Pipeline Example
--------------------------------

This example shows the utilization of a ``pmdarima.pipeline.Pipeline``, incorporating preprocessing operations
to each series. In the example below, a
`Box Cox <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.preprocessing.BoxCoxEndogTransformer.html>`_
transformation is applied to each series to force stationarity.

.. note::
    The data set used for these examples is a randomly generated non-deterministic group of series data. As such,
    The relevance of utilizing a normalcy transform on this data is somewhere between 'unlikely' and 'zero'.
    Using a BoxCox transform here is used as an API example only.

.. literalinclude:: /../examples/grouped_pmdarima/pipeline_example.py
    :caption: GroupedPmdarima with Pipeline model
    :language: python
    :linenos:

GroupedPmdarima Group Subset Prediction Example
-----------------------------------------------

This example shows a subset prediction of groups by using the `predict_groups <diviner.GroupedPmdarima.predict_groups>`
method.

.. literalinclude:: /../examples/grouped_pmdarima/group_subset_arima_example.py
    :caption: GroupedPmdarima Subset Groups Prediction
    :language: python
    :linenos:

GroupedPmdarima Series Analysis Example
---------------------------------------

The below script illustrates how to perform analytics on a grouped series data set. Applying the results of these
utilities can aid in determining appropriate order values (p, d, q) and seasonal order values (P, D, Q) for the
example shown in :ref:`the ARIMA example <arima_script>`.

.. literalinclude:: /../examples/grouped_pmdarima/grouped_series_exploration.py
    :caption: GroupedPmdarima series exploration and analysis
    :language: python
    :linenos:

GroupedPmdarima Differencing Term Manual Calculation Example
------------------------------------------------------------

This script below shows a means of dramatically reducing the optimization time of AutoARIMA through the manual
calculation of the differencing term ``'d'`` for each series in the grouped series data set. By manually setting
this argument (which can be either unique for each group or homogenous across all groups), the optimization algorithm
can reduce the total number of iterative validation tests.

.. literalinclude:: /../examples/grouped_pmdarima/pmdarima_analyze_differencing_terms_and_apply.py
    :caption: GroupedPmdarima manual differencing term extraction and application to AutoARIMA
    :language: python
    :linenos:

Supplementary
-------------

.. note:: To run these examples for yourself with the data generator example, utlize the following code:

.. literalinclude:: /../examples/example_data_generator.py
    :caption: Synthetic Data Generator
    :language: python
    :linenos:
