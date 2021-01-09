#!/usr/bin/env python

# python setup.py sdist
# pip install -e .

# https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html


from distutils.core import setup

setup(name='pystatplottools',
      version='0.1',
      description='Python modules for performing simple statistics and plotting routines',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['pystatplottools',
                'pystatplottools.distributions',
                'pystatplottools.expectation_values'
                'pystatplottools.pdf_env',
                'pystatplottools.plotting',
                'pystatplottools.pytorch_data_generation',
                'pystatplottools.utils',
                'pystatplottools.visualization']
     )
