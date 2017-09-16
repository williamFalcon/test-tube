#!/usr/bin/env python
from setuptools import setup, find_packages

version = '0.32'

setup(name='test_tube',
      packages=find_packages(),
      version=version,
      description='Experiment logger and visualizer',
      author='William Falcon',
      install_requires=['numpy==1.12.1'],
      author_email='will@hacstudios.com',
      url='https://github.com/williamFalcon/test_tube',
      download_url='https://github.com/williamFalcon/test_tube/archive/{}.tar.gz'.format(version),
      keywords=['testing', 'machine learning', 'deep learning', 'prototyping', 'experimenting', 'modeling']
     )
