# content of setup.py
from setuptools import setup, find_packages

setup(
    name="moist_thermodynamics",
    version="0.3",
    description="Constants and functions for the treatment of moist atmospheric thermodynamics",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
)
