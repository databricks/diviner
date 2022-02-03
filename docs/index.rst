.. Diviner documentation master file, created by
   sphinx-quickstart on Thu Dec 30 12:18:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diviner Documentation
===================================
Diviner is an open source library for large-scale (multi-series) time series forecasting.
It serves as a wrapper around popular open source forecasting libraries, providing a consolidated
framework for simplified modeling of many discrete series components with an efficient high-level API.

To get started with Diviner, see the quickstart guide (:ref:`quickstart`).

Individual forecasting library API guides can be found here:

* :ref:`Grouped Prophet API <grouped_prophet>`
* :ref:`Grouped Pmdarima API <grouped_pmdarima>`

Examples of each of the wrapped forecasting libraries can be found :ref:`here<tutorials-and-examples>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   quickstart
   tutorials-and-examples/index
   grouped_prophet
   grouped_pmdarima
   data_processing
   contributing
   changelog

