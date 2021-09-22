from distutils.core import setup
from setuptools import find_packages

setup(name='mutex',
      version='1.0.1',
      description='A simple algorithm to estimate the significance of a mutual exclusive pattern or co-occurrence',
      author='Michael P Schroeder',
      author_email='michael.p.schroeder@gmail.com',
      url='https://github.com/mpschr/mutex',
      packages=find_packages(),
      install_requires=['pandas>=0.17', 'numpy>=1.10.4', 'joblib', 'multiprocessing', 'scipy>=0.17']
      )
