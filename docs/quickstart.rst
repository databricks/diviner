.. _quickstart:

Quickstart
==========

This guide will walk through the basic elements of Diviner's API, discuss the approach to data structures, and show a
simple example of using a ``GroupedProphet`` model to generate forecasts for a multi-series data set.

.. contents:: Table of Contents
    :local:
    :depth: 1

Installing Diviner
------------------
Diviner can be installed by running:

.. code-block:: python

    pip install diviner

.. note::
    Diviner's underlying wrapped libraries have requirements that may require additional environment configuration,
    depending on your operating system's build environment. Libraries such as `Prophet <https://facebook.github.io/prophet/>`_
    require gcc compilation libraries to build the underlying solvers that are used (PyStan). As such, it is recommended to use
    a version of Python >= 3.8 to ensure that the build environment resolves correctly and to ensure that the system's pip version is updated.

Downloading Examples
--------------------
Examples can be viewed `here <https://github.com/databricks/diviner/tree/main/examples>`_ for each supported forecasting
library that is part of Diviner.
To see examples of how to use Diviner to perform multi-series forecasting, download the examples by cloning the Diviner
repository via:

.. code-block:: console

    git clone https://github.com/databricks/diviner

Once cloned, ``cd`` into ``<root>/diviner/examples`` to find both scripts and Jupyter notebook examples for each
of the underlying wrapped forecasting libraries available. Running these examples will auto-generate a data set,
generate the names of columns that define the groupings within the data set, and generate grouped forecasts for each
of the defined groups in the data set.

Input Dataset Structure
-----------------------
The Diviner APIs use a DataFrame-driven approach to define and process distinct univariate time series data.
The principle concept in use to accomplish the definition of discrete series information is `stacking`, wherein
there is a single column that defines the endogenous regression term (a 'y value' that represents the data that is to be forecast),
a column that contains the datetime (or date) values, and identifier column(s) that describe the distinct series data
within the datetime and endogenous columns.

As an example, let's look at a denormalized data set that is typical in other types of machine learning.
The data shown below is typically the result of transformation logic from the underlying stored data in a Data Warehouse,
rather than how data such as sales figures would be stored in fact tables.

.. list-table:: Denormalized Data
    :widths: 25 25 25 25 25
    :header-rows: 1

    * - Date
      - London Sales
      - Paris Sales
      - Munich Sales
      - Rome Sales
    * - 2021-01-01
      - 154.9
      - 83.9
      - 113.2
      - 17.5
    * - 2021-01-02
      - 172.4
      - 91.2
      - 101.7
      - 13.4
    * - 2021-01-03
      - 191.2
      - 87.8
      - 99.9
      - 18.1
    * - 2021-01-04
      - 155.5
      - 82.9
      - 127.8
      - 19.2
    * - 2021-01-05
      - 168.4
      - 92.8
      - 104.4
      - 9.6

This storage paradigm, while useful for creating a feature vector for supervised machine learning, is not
efficient for large-scale forecasting of univariate time series data. Instead of requiring data in this format, Diviner
requires a 'normalized' (stacked) data structure similar to how data would be stored in a database.
Below is the representation of the same data, restructured how Diviner requires it to be structured.

.. list-table:: Stacked Data (Format for Diviner)
    :widths: 30 30 30 30
    :header-rows: 1

    * - Date
      - Country
      - City
      - Sales
    * - 2021-01-01
      - United Kingdom
      - London
      - 154.9
    * - 2021-01-02
      - United Kingdom
      - London
      - 172.4
    * - 2021-01-03
      - United Kingdom
      - London
      - 191.2
    * - 2021-01-04
      - United Kingdom
      - London
      - 155.5
    * - 2021-01-05
      - United Kingdom
      - London
      - 168.4
    * - 2021-01-01
      - France
      - Paris
      - 83.9
    * - 2021-01-02
      - France
      - Paris
      - 91.2
    * - 2021-01-03
      - France
      - Paris
      - 87.8
    * - 2021-01-04
      - France
      - Paris
      - 82.9
    * - 2021-01-05
      - France
      - Paris
      - 92.8
    * - 2021-01-01
      - Germany
      - Munich
      - 113.2
    * - 2021-01-02
      - Germany
      - Munich
      - 101.7
    * - 2021-01-03
      - Germany
      - Munich
      - 99.9
    * - 2021-01-04
      - Germany
      - Munich
      - 127.8
    * - 2021-01-05
      - Germany
      - Munich
      - 104.4
    * - 2021-01-01
      - Italy
      - Rome
      - 17.5
    * - 2021-01-02
      - Italy
      - Rome
      - 13.4
    * - 2021-01-03
      - Italy
      - Rome
      - 18.1
    * - 2021-01-04
      - Italy
      - Rome
      - 19.2
    * - 2021-01-05
      - Italy
      - Rome
      - 9.6

This data structure paradigm enables several utilization paths that a 'pivoted' data structure, as shown above in the 'Denormalized Data'
example, does not, such as:

* The ability to dynamically group data based on hierarchical relationships.
    * Group on {"Country", "City"}
    * Group on {"Country"}
    * Group on {"City"}
* Less data manipulation transformation code required when pulling data from source systems.
* Increased legibility of visual representations of the data.

Basic Example
-------------
To illustrate how to build forecasts for our country sales data above, here is an example of building a grouped
forecast for each of the cities using the :ref:`GroupedProphet API <grouped_prophet>`.

.. code-block:: python

    import pandas as pd
    from diviner import GroupedProphet

    series_data = pd.read_csv("/data/countries")

    grouping_columns = ["Country", "City"]

    grouped_prophet_model = GroupedProphet().fit(
        df=series_data,
        group_key_columns=grouping_columns
    )

    forecast_data = grouped_prophet_model.forecast(horizon=30, frequency="D")

This example will parse the columns "Country" and "City", generate grouping keys, and build Prophet models for
each of the combinations present in the data set:

{("United Kingdom", "London"), ("France", "Paris"), ("Germany", "Munich"), ("Italy", "Rome")}

Alternatively, if we had multiple city values for each country and wished to forecast sales by country, we could have
submitted ``grouping_columns = ["Country"]`` and the data would have been aggregated to the country level, building models that
would have been at the country level.

Following the model building, a 30 day forecast was generated (returned as a stacked consolidated Pandas DataFrame).

.. note::
    For more in-depth examples (including per-group parameter extraction, cross validation metrics results, and serialization),
    see the :ref:`examples <tutorials-and-examples>`.