import os

from setuptools import setup

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

setup(package_data={'': package_files('martinize2/mapping'),},
      setup_requires=['setuptools>=30.3.0'])

