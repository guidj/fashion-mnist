from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'tensorflow==2.0.1',
    'Keras==2.3.0',
    'opencv-python==4.2.0.32',
    'scikit-learn==0.21.3',
    'gcsfs==0.3.1',
    'hyperopt==0.2.2',
    'pyspark==2.4.4',
    'cloudpickle==1.2.2'
]

setup(
    name='fmnist',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='guilherme',
    license='MIT',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False
)
