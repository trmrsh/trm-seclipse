from setuptools import setup, Extension
import os, numpy
from codecs import open
from os import path
from Cython.Build import cythonize

""" Setup script for the eclipsing spheres python extension"""

#seclipse = [Extension(
#    'trm.seclipse.seclipse',
#    [os.path.join('trm','seclipse','seclipse.pyx')],
#    define_macros   = [('MAJOR_VERSION', '0'),
#                       ('MINOR_VERSION', '1')])]

setup(name='trm.seclipse',
      version     = '1',
      packages    = ['trm', 'trm.seclipse',],
#      ext_modules=cythonize(seclipse),

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python sphere eclipse module",
      url='http://www.astro.warwick.ac.uk/',
      )

