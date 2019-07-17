#!/usr/bin/env python
from setuptools import find_packages, setup

version = '0.6.7.1'

setup(
    name='test_tube',
    packages=find_packages(),
    version=version,
    description='Experiment logger and visualizer',
    author='William Falcon',
    install_requires=[
        'pandas>=0.20.3',
        'numpy>=1.13.3',
        'imageio>=2.3.0',
        'tb-nightly==1.15.0a20190708',
        'torch>=1.1.0',
        'future'
    ],
    author_email='will@hacstudios.com',
    url='https://github.com/williamFalcon/test_tube',
    download_url='https://github.com/williamFalcon/test_tube/archive/{}.tar.gz'.format(version),
    keywords=[
        'testing',
        'machine learning',
        'deep learning',
        'prototyping',
        'experimenting',
        'modeling',
    ],
)
