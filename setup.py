#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
import os

version = '0.1'
readme = open('README.txt').read()

setup(name='test_tube',
      packages=['test_tube'],
      version=version,
      description='Experiment logger and visualizer',
      long_description=readme,
      author='William Falcon',
      author_email='will@hacstudios.com',
      url='https://github.com/williamFalcon/test_tube',
      download_url='https://github.com/williamFalcon/test_tube/archive/0.1.tar.gz',
      keywords=['testing', 'machine learning', 'deep learning', 'prototyping', 'experimenting', 'modeling']
     )
