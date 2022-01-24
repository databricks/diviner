Data Processing
===============

.. toctree::
    :maxdepth: 2

The data manipulation APIs are a key component of the utility of this library. While they are largely obfuscated by the
main entry point APIs for each forecasting library, they can be useful for custom implementations and for performing
validation of source data sets.

Pandas DataFrame Group Processing API
-------------------------------------

Class signature:

.. autoclass:: diviner.data.pandas_group_generator.PandasGroupGenerator
    :members:
    :private-members:
    :special-members:

Developer API for Data Processing
---------------------------------

Abstract Base Class for grouped processing of a fully normalized ``DataFrame`` :

.. automodule:: diviner.data.base_group_generator
    :members:
    :private-members:
    :special-members:
