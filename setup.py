#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
import os

version = '0.24'

setup(name='test_tube',
      packages=['test_tube'],
      version=version,
      description='Experiment logger and visualizer',
      author='William Falcon',
      author_email='will@hacstudios.com',
      url='https://github.com/williamFalcon/test_tube',
      download_url='https://github.com/williamFalcon/test_tube/archive/{}.tar.gz'.format(version),
      keywords=['testing', 'machine learning', 'deep learning', 'prototyping', 'experimenting', 'modeling']
     )
