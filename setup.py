#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(name='test_tube',
      packages=['test_tube'],
      version='0.1',
      description='Experiment logger and visualizer',
      author='William Falcon',
      author_email='will@hacstudios.com',
      url='https://github.com/williamFalcon/test_tube',
      download_url='https://github.com/williamFalcon/test_tube/archive/0.1.tar.gz',
      install_requires=required,
      keywords=['testing', 'machine learning', 'deep learning', 'prototyping', 'experimenting', 'modeling']
     )
