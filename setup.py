import os
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

version = (
    SourceFileLoader("diviner.version", os.path.join("diviner", "version.py"))
    .load_module()
    .VERSION
)

REQUIREMENTS = ["numpy", "pandas", "prophet", "pmdarima"]

setup(
    name="diviner",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIREMENTS,
    zip_safe=False,
    author="Databricks",
    description="Diviner: A Grouped Forecasting API",
    license="Apache2.0",
    classfiers=[
        "Intended Audience: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="ml ai databricks",
    python_requires=">=3.7",
)
