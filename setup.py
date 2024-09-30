from setuptools import setup, find_packages

setup(
    name='poisoning_bayesian_inference',
    version='0.1',
    packages=find_packages(),
    package_data={
        'poisoning_bayesian_inference': ['data/*.csv'],  # Include CSV files in the data directory
    },
    include_package_data=True
)