
Diviner: Grouped Timeseries Forecasting at scale
================================================

Diviner is an execution framework wrapper around popular open source time series forecasting libraries.
The aim of the project is to simplify the creation, training, orchestration, and MLOps logistics associated with
forecasting projects that involve the predictions of many discrete independent events.

|docs| |pypi| |license| |downloads|

.. image:: https://readthedocs.org/projects/databricks-diviner/badge/?version=latest
    :target: https://databricks-diviner.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation 

.. |pypi| image:: https://img.shields.io/badge/pypi/v/diviner.svg
    :target: https://pypi.org/project/diviner/
    :alt: Latest Python Release

.. |license| image:: https://img.shields.io/badge/license-Apache%202-brightgreen.svg
    :target: https://github.com/databricks/diviner/blob/main/LICENSE.txt
    :alt: Apache 2 License

.. |downloads| image:: https://pepy.tech/badge/diviner
    :target: https://pepy.tech/project/diviner
    :alt: Total Downloads


Is this right for my project?
-----------------------------

Diviner is meant to help with large-scale forecasting. Instead of describing each individual use case where it may be
applicable, here is a non-exhaustive list of projects that it would fit well as a solution for:

* Forecasting regional sales within each country that a company does business in per day
* Predicting inventory demand at regional warehouses for thousands of products
* Forecasting traveler counts at each airport within a country daily
* Predicting electrical demand per neighborhood (or household) in a multi-state region

Each of these examples has a *common theme*:

* The data is temporally homogenous (all of the data is collected daily, hourly, weekly, etc.).
* There is a large number of individual models that need to be built due to the cardinality of the data.
* There is no guarantee of seasonal, trend, or residual homogeneity in each series.
* Varying levels of aggregation may be called for to solve different use cases.

The primary aspect that Diviner solves is allowing for managing of the execution of many discrete series with a
high-level API and metadata management approach that relieves the operational burden of managing hundreds (or thousands)
of individual models.

If a project needs to provide forecasts for many entities utilizing common open source libraries, Diviner will provide
a simpler solution to build these projects than having to write an infrastructure framework to handle individual model
instances.

Grouped Modeling Wrappers
-------------------------

Currently, Diviner supports the following open source libraries for forecasting at scale:

* `prophet <https://facebook.github.io/prophet/docs/quick_start.html>`_

* `statsmodels <https://www.statsmodels.org/stable/index.html>`_

* `pmdarima <https://alkaline-ml.com/pmdarima/index.html>`_

Installing
----------

Install Diviner from PyPi via ``pip install diviner``

Documentation
-------------

Documentation, Examples, and Tutorials for Diviner can be found `here <https://databricks-diviner.readthedocs.io/en/latest/index.html>`_.

Community & Contributing
------------------------

For assistance with Diviner, see the `docs <https://databricks-diviner.readthedocs.io/en/latest/index.html>`_.

Contributions to Diviner are welcome. To file a bug, request a new feature, or to contribute a feature request, please
open a GitHub issue. The team will work with you to ensure that your contributions are evaluated and appropriate
feedback is provided. See :ref:`contributing guidelines <../CONTRIBUTING.md>`_.
