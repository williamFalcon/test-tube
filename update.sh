#!/bin/bash

version=$1

git commit -am "release v$version"
git tag $version -m "test_tube v$version"
git push --tags origin master

python setup.py sdist upload -r pypi