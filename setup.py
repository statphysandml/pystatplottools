#!/usr/bin/env python

# python setup.py sdist
# pip install -e .

# https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html


from distutils.core import setup

setup(name='pystatplottools',
      version='1.0',
      description='Python modules for performing simple statistics and plotting routines',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['pystatplottools', 'pystatplottools.ppd_distributions',
                'pystatplottools.ppd_pdf_env', 'pystatplottools.ppd_plotting_env', 'pystatplottools.ppd_loading']
     )