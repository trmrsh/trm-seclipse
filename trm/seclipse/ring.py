"""Set of routines for calculating mutual eclipses between 3 and 4
limb-darkened spheres based on splitting them up into a series of annuli
centred on the visible face. Each annulus of a given sphere has a constant
surface brightness and so the visible flux can be calculated by calculating
the geometry of its occultation behind the other sphere. This makes the
problem 1D albeit with various trig function calls over loops needed.

For an alternative approach see seclipse.grid.

Many of the routines in this section use Cython for speed.
"""

import numpy as np
import math
from ._seclipse import lc3, lc4, flux2, flux3, flux4

def rfinit(r, limb, n):
    """Initialises radii and fluxes from set of annuli covering the visible face
    of a star of unit central surface brightness.

    Arguments::

      r : (float)
         stellar radius

      limb : (Limb)
         limb darkening

      n : (int)
         number of annuli (should be at least 2).

    Returns (rings, fluxes, tflux) where::

      rings  : (array)
         ring radii to use when computing obscuration

      fluxes : (array)
         unocculted contributions of each ring to the total flux, which
         includes trapezoidal weighting.

      tflux  : (float)
         total flux, i.e. sum over fluxes.

    """

    # Generate radii of annuli and trapezoidal weights.  The radii start at 0
    # end at r, except we drop the zero one from the start since it does not
    # contribute to the final summation, leaving only the outermost one with a
    # half weight.
    rings = np.linspace( r/n, r, n)
    w = np.ones_like(rings)
    w[-1] = 0.5

    # compute mu values
    mu = np.sqrt(1-(rings/r)**2)

    # limb darkenings
    lmb = limb(mu)

    # fluxes
    fluxes = (2.*math.pi*r/n)*(w*rings*lmb)

    # return the results
    return (rings, fluxes, fluxes.sum())

