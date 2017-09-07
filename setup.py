#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(name='ngv_ds_utils',
      version='0.0',
      description='NGV DS Utils',
      author='William Falcon',
      author_email='will@nextgenvest.com',
      url='',
      packages=['ngv_ds'],
      install_requires=required
     )
