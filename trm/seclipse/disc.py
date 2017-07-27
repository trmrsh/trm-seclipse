"""Routines for calculating the eclipse of an inclined circular disc
by spheres. It does this by splitting the disc into roughly equal-sized
elements.
"""

import sys
import math
import numpy as np
import numexpr as ne
import trm.subs as subs
from trm.subs import Vec3
from . import ring

def rfinit(r, bright, n):
    """Initialises the grid elements and flux contributions over the face of a
    disc using a series of n annuli covering its face.

    Arguments::

      r      : (float)
         outer disc radius radius

      bright : (callable)
         function taking one argument representing the radius scaled as a
         fraction of the outer radius, i.e. from 0 to 1 inclusive, returning
         the surface brightness.

      n      : (int)
         number of annuli (should be at least 2).

    Returns (x, y, fluxes, tflux) where::

      x  : (array)
         x locations of elements

      y  : (array)
         y locations of elements

      fluxes : (array)
         unocculted contributions of each element to the total flux

      tflux  : (float)
         total flux, i.e. sum over fluxes.
    """

    # Generate the radii of the annuli. These should be regarded as being at
    # the # mid-points radially of each annulus.
    halfdr = r/n/2
    radii = np.linspace(halfdr, r-halfdr, n)

    # initially count number of elements needed so arrays can be setup
    ngrid = 0
    for radius in radii:
        ngrid += max(8, int(math.pi*radius/halfdr))
    x = np.empty(ngrid)
    y = np.empty(ngrid)
    f = np.empty(ngrid)

    ngrid = 0
    for radius in radii:
        # compute number of thetas to make elements roughly square
        # must be same as the equivalent line in the previous loop
        ntheta = max(8, int(math.pi*radius/halfdr))

        # compute x, y [on sky] coordinates
        thetas = np.linspace(0,2.*math.pi*(1-1/ntheta),ntheta)
        xa = radius*np.cos(thetas)
        ya = radius*np.sin(thetas)

        # area of elements in this annulus
        area = radius*(2.*math.pi/ntheta)*(r/n)

        # now the fluxes
        fa = area*bright(radius/r)*np.ones(ntheta)

        # store values just calculated
        x[ngrid:ngrid+ntheta] = xa
        y[ngrid:ngrid+ntheta] = ya
        f[ngrid:ngrid+ntheta] = fa

        # update offset pointer
        ngrid += ntheta

    return (x, y, f, f.sum())

def project(x, y, iangle, OMEGA):
    """
    Given the x, y coordinates of elements covering a disc face as produced by
    rfinit (i.e. x, y coordinates in the plane of the disc), this routine
    projects them onto the plane of the sky assuming that the disc has orbital
    inclination iangle (radians) and longitude of line of nodes OMEGA
    (radians, measured anti-clockwise from North on the sky).

    Sky axes: due West = x, due North = y, towards Earth = z.
    """
    # express axes referenced with respect to the disc in terms of axes
    # referenced with respect to the sky. oaxis is the orbital / symmetry axis
    # of the disc, naxis is the axis of the lines of nodes (by definition in
    # the plane of the sky) while paxis [p for perpendicular] is such that
    # naxis / paxis / oaxis are a right-handed Cartesian set, i.e. paxis =
    # oaxis x naxis [cross product]
    #
    # The output positions are then given by naxis*x + paxis*y and we are only interested
    # in the x and y components.
    naxis = Vec3(-math.sin(OMEGA), math.cos(OMEGA), 0)
    oaxis = Vec3(-math.sin(iangle)*math.cos(OMEGA), -math.sin(iangle)*math.sin(OMEGA), math.cos(iangle))
    paxis = subs.cross(oaxis, naxis)

    return (x*naxis.x + y*paxis.x, x*naxis.y + y*paxis.y)

def flux2(x, y, f, tf, r1, x1, y1, z1, r2, x2, y2, z2):
    """Computes the flux from a disc potentially occulted by 2 spheres. The x and y arrays
    here are best computed from "rfinit" followed by "project".

    Arguments::

      x   : (array)
         X-ordinates of disc elements on plane of sky

      y   : (array)
         Y-ordinates of disc elements on plane of sky

      f   : (array)
         Flux contributions of disc elements

      tf  : (float)
         Total flux from disc.

      r1  : (float)
         radius of first sphere

      x1  : (float)
         X position of first sphere relative to centre of disc

      y1  : (float)
         Y position of first sphere relative to centre of disc

      z1  : (float)
         Z position of first sphere relative to centre of disc

      r2  : (float)
         radius of second sphere

      x2  : (float)
         X position of second sphere relative to centre of disc

      y2  : (float)
         Y position of second sphere relative to centre of disc

      z2  : (float)
         Z position of second sphere relative to centre of disc. Positive
         Z is towards Earth.

    Returns a single number, the total visible flux from the disc.

    """

    if z1 > 0 and z2 > 0:
        # both spheres potentially can eclipse disc
        flux = f[((x-x1)**2 + (y-y1)**2 > r1**2) & ((x-x2)**2 + (y-y2)**2 > r2**2)].sum()
    elif z1 > 0:
        # only sphere 1 can occult the disc
        flux = f[(x-x1)**2 + (y-y1)**2 > r1**2].sum()
    elif z2 > 0:
        # only sphere 2 can occult the disc
        flux = f[(x-x2)**2 + (y-y2)**2 > r2**2].sum()
    else:
        # no occultation of disc
        flux = tf
    return flux

def lc2(x, y, f, tf,
        r1, rings1, fluxes1, tflux1, s1, x1s, y1s, z1s,
        r2, rings2, fluxes2, tflux2, s2, x2s, y2s, z2s):
    """"Wrapper around 'flux2' to run over lots of positions specified in the
    x1s, y1s etc which are array versions of the same parameters in flux2
    """

    n = len(x1s)
    lc = np.empty(n)

    # build tuples for ring.flux2
    r = (r1,r2)
    rings = (rings1,rings2)
    fluxes = (fluxes1,fluxes2)
    tflux = (tflux1,tflux2)

    for i in range(n):
        # compute what we see from the disc
        fdisc = flux2(x, y, f, tf, r1, x1s[i], y1s[i], z1s[i], r2, x2s[i], y2s[i], z2s[i])

        # compute what we see from the two stars. we ignore the
        # possibility of the disc occulting the stars.
        f1,f2 = ring.flux2(r, rings, fluxes, tflux, (x1s[i],x2s[i]), (y1s[i],y2s[i]), (z1s[i],z2s[i]))

        # Multiply in surface brightnesses to get final number.
        lc[i] = s1*f1 + s2*f2 + fdisc

    return lc
