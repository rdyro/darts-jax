import os
from pathlib import Path
from setuptools import setup

setup(
    name="jfi",
    version="0.3.2",
    author="Robert Dyro",
    description=("Simplified and user friendly interface to JAX."),
    license="MIT",
    packages=["jfi"],
    long_description=(Path(__file__).absolute().parent / "README.md").read_text(),
)
