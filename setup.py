#!/usr/bin/env python
import os
from setuptools import find_packages, setup

version = '0.7.4'
PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


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
        'tensorboard>=1.15.0',
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
