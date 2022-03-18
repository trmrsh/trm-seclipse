from setuptools import setup, find_packages
from setuptools.extension import Extension
import os, numpy
from os import path
from Cython.Build import cythonize

""" Setup script for the eclipsing spheres python extension"""

seclipse = [Extension(
    'trm.seclipse._seclipse',
    [os.path.join('trm','seclipse','_seclipse.pyx')],
#    libraries=["m"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fno-strict-aliasing"],
    define_macros   = [('MAJOR_VERSION', '0'),
                       ('MINOR_VERSION', '1')]),
            ]

setup(name='trm.seclipse',
      version = '1.0.0',
      packages=find_packages(exclude=['docs', 'tests']),
      ext_modules=cythonize(seclipse),

      entry_points = {
          'console_scripts' : [
#              'cpaths=trm.seclipse.scripts.cpaths:cpaths',
              'lcmodel=trm.seclipse.scripts.lcmodel:lcmodel',
#              'masses=trm.seclipse.scripts.masses:masses',
              'mcmc=trm.seclipse.scripts.mcmc:mcmc',
#              'ppath=trm.seclipse.scripts.ppath:ppath',
              'ppxy=trm.seclipse.scripts.ppxy:ppxy',
              'ppxz=trm.seclipse.scripts.ppxz:ppxz',
              'ppxyz=trm.seclipse.scripts.ppxyz:ppxyz',
          ]
      },

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python sphere eclipse module",
      url='http://www.astro.warwick.ac.uk/',
      )

