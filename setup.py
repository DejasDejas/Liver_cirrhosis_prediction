from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='use xgboost to predict liver cirrhosis in a patient based on certain lifestyle and health conditions '
                'of a patients.',
    author='Julien Dejasmin',
    license='MIT',
)
