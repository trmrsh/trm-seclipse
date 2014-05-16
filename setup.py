from distutils.core import setup, Extension
import os, numpy

""" Setup script for the atomic python extension"""


setup(name='trm.seclipse',
      version     = '0.9',
      packages    = ['trm', 'trm.seclipse',],

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python sphere eclipse module",
      url='http://www.astro.warwick.ac.uk/',
      )

