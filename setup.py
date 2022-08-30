from setuptools import setup, find_packages
import pathlib
from typing import Union, List


def get_version(rel_path: Union[List[str], str]) -> str:
    if isinstance(rel_path, str):
        rel_path = [rel_path]
    read_path = pathlib.Path(*rel_path).absolute()
    for line in read_path.read_text().splitlines():
        if line.startswith("VERSION"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


REQUIREMENTS = ["numpy", "pandas", "prophet", "pmdarima", "packaging"]

setup(
    name="diviner",
    version=get_version(["diviner", "version.py"]),
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIREMENTS,
    zip_safe=False,
    author="Databricks",
    author_email="benjamin.wilson@databricks.com",
    url="https://github.com/databricks/diviner",
    project_urls={
        "Issue Tracker": "https://github.com/databricks/diviner/issues",
        "Documentation": "https://databricks-diviner.readthedocs.io/en/latest/",
    },
    description="Diviner: A Grouped Forecasting API",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="Apache2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 3 - Alpha",
    ],
    keywords="ml ai forecasting databricks",
    python_requires=">=3.7",
)
