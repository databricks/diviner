GroupedProphet Example Scripts
==============================

The scripts included below show the various options available for utilizing the ``GroupedProphet`` API.
For an alternative view of these examples with data visualizations, see the :ref:`notebooks here <tutorials-and-examples>`

.. contents:: Scripts
    :local:
    :depth: 2

GroupedProphet Example
----------------------

This script shows a simple example of training a series of Prophet models, saving a ``GroupedProphet`` instance,
loading that instance, cross validating through backtesting, and generating a forecast for each group.

.. literalinclude:: /../examples/grouped_prophet/grouped_prophet_example.py
    :caption: GroupedProphet Script
    :language: python
    :linenos:

GroupedProphet Subset Group Prediction Example
----------------------------------------------

This script shows a simple example of training a series of Prophet models and generating a group subset prediction.

.. literalinclude:: /../examples/grouped_prophet/group_subset_prophet_example.py
    :caption: GroupedProphet Subset Groups Script
    :language: python
    :linenos:

Supplementary
-------------

.. note:: To run these examples for yourself with the data generator example, utlize the following code:

.. literalinclude:: /../examples/example_data_generator.py
    :caption: Synthetic Data Generator
    :language: python
    :linenos:
