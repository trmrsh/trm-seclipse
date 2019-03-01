"""
Scripts sub-module of seclipse contains the commands to runs models etc
"""

from .lcmodel import lcmodel
from .mcmc import mcmc

__all__ = [ \
            'lcmodel',
            'mcmc',
        ]

