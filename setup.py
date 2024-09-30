from setuptools import setup, find_packages

setup(
    name='Poisoning_Bayesian_Inference',
    version='0.1',
    packages=find_packages(),
    package_data={
        'Poisoning_Bayesian_Inference': ['data/*.csv'],  # Include CSV files in the data directory
    },
    include_package_data=True
)