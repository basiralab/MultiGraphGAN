"""
Copyright (c) 2017, Stanford University.
All rights reserved.

This source code is a Python implementation of SIMLR for the following paper published in Nature Methods:
Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning
"""
from distutils.core import setup

setup(
    name='SIMLR',
    version='0.1.3',
    author='Bo Wang',
    author_email='bowang87@stanford.edu',
    url='https://github.com/bowang87/SIMLR-PY',
    description='Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning',
    packages=['SIMLR'],
    install_requires=[
                   'fbpca>=1.0',
                   'numpy>=1.8.0',
                   'scipy>=0.13.2',
                   'annoy>=1.8.0',
                   'scikit-learn>=0.17.1',
     ],
    classifiers=[])
