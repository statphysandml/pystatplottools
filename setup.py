#!/usr/bin/env python

# python setup.py sdist
# pip install -e .

# https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

from distutils.core import setup

setup(name='pystatplottools',
      version='0.1',
      description='Python modules for performing simple statistics and plotting routines',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      url='https://github.com/statphysandml/pystatplottools',
      packages=['pystatplottools',
                'pystatplottools.distributions',
                'pystatplottools.expectation_values',
                'pystatplottools.pdf_env',
                'pystatplottools.plotting',
                'pystatplottools.pytorch_data_generation',
                'pystatplottools.utils',
                'pystatplottools.visualization'],
      long_description=long_description,
      long_description_content_type='text/markdown'
     )
