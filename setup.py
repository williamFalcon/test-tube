#!/usr/bin/env python
from setuptools import setup, find_packages

version = '0.38'

setup(name='test_tube',
      packages=find_packages(),
      version=version,
      description='Experiment logger and visualizer',
      author='William Falcon',
      install_requires=['scikit-image>=0.12.3'],
      author_email='will@hacstudios.com',
      url='https://github.com/williamFalcon/test_tube',
      download_url='https://github.com/williamFalcon/test_tube/archive/{}.tar.gz'.format(version),
      keywords=['testing', 'machine learning', 'deep learning', 'prototyping', 'experimenting', 'modeling']
     )

"""
To upload change:

git commit -am "updated setup.py"  
git tag 0.38 -m "test_tube v0.38"  
git push --tags origin master   

python setup.py sdist upload -r pypi
"""