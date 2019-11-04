from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'tensorflow == 1.14.0',
    'Keras == 2.3.0',
    'opencv-python==4.1.1.26',
    'scikit-learn==0.21.3',
    'gcsfs==0.3.1'
]

setup(
    name='fmnist',
    packages=find_packages(),
    # packages=['fmnist',
    #           'fmnist.data', 'fmnist.features', 'fmnist.models',
    #           'fmnist.visualization'],
    version='0.1.0',
    description='A short description of the project.',
    author='guilherme',
    license='MIT',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False
)
