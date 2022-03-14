from setuptools import setup, find_packages
import os


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


REQUIREMENTS = ["numpy", "pandas", "prophet", "pmdarima"]

setup(
    name="diviner",
    version=get_version("diviner/__init__.py"),
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    install_requires=REQUIREMENTS,
    zip_safe=False,
    author="Databricks",
    author_email="benjamin.wilson@databricks.com",
    url="http://databricks-diviner.readthedocs.io/",
    description="Diviner: A Grouped Forecasting API",
    license="Apache2.0",
    classifiers=[
        "Intended Audience: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="ml ai databricks",
    python_requires=">=3.7",
)
