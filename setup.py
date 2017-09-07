#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(name='test_tube',
      version='0.0',
      description='Experiment logger and visualizer',
      author='William Falcon',
      author_email='will@hacstudios.com',
      url='',
      packages=['log'],
      install_requires=required
     )
