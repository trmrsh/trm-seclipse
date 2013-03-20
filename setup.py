from distutils.core import setup, Extension
import os, numpy

""" Setup script for the atomic python extension"""

#library_dirs = []
#include_dirs = []

# need to direct to where includes and  libraries are
#if os.environ.has_key('TRM_SOFTWARE'):
#    library_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'lib'))
#    include_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'include'))
#else:
#    print >>sys.stderr, "Environment variable TRM_SOFTWARE pointing to location of shareable libraries and includes not defined!"

#include_dirs.append(numpy.get_include())

#atomic = Extension('trm.atomic._atomic',
#                define_macros   = [('MAJOR_VERSION', '0'),
#                                   ('MINOR_VERSION', '1')],
#                undef_macros    = ['USE_NUMARRAY'],
#                include_dirs    = include_dirs,
#                library_dirs    = library_dirs,
#                runtime_library_dirs = library_dirs,
#                libraries       = ['subs'],
#                sources         = [os.path.join('trm', 'atomic', 'atomic.cc')])

setup(name='trm.seclipse',
      version     = '0.9',
      packages    = ['trm', 'trm.seclipse',],
#      ext_modules = [seclipse],

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python sphere eclipse module",
      url='http://www.astro.warwick.ac.uk/',
      )

