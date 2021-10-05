"""Routines for calculating the eclipse of an inclined circular disc
by spheres. It does this by splitting the disc into roughly equal-sized
elements.
"""

import sys
import math
import numpy as np
import numexpr as ne
from trm.vec3 import Vec3
from . import ring
from . import ring
from ._seclipse import dflux2

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


def d2s_axes(iangle, OMEGA):
    """
    Returns x,y,z basis vectors defined by the disc in terms of the sky axes.
    x of the disc is in the plane of the sky along the line of nodes. If
    OMEGA=0 it points North. The z-axis is defined to be perpendicular to the
    the disc and towards the observer for iangle < 90. The y-axis is defined
    so that a right-handed Cartesian triad emerges.  iangle is the angle
    between the disc's z axis and the line of sight.

    The sky axes are defined by due West = x, due North = y, towards Earth (line of sight)
    = z.

    This returns a tuple of Vec3 objects (xd,yd,zd)
    """
    coso, sino = math.cos(OMEGA), math.sin(OMEGA)
    cosi, sini = math.cos(iangle), math.sin(iangle)

    xd = Vec3(-sino, coso, 0)
    yd = Vec3(-cosi*coso, -cosi*sino, -sini)
    zd = Vec3(-sini*coso, -sini*sino,  cosi)

    return (xd,yd,zd)

def project(x, y, iangle, OMEGA):
    """
    Given the x, y coordinates of elements covering a disc face as produced by
    rfinit (i.e. x, y coordinates in the plane of the disc), this routine
    projects them onto the plane of the sky assuming that the disc has orbital
    inclination iangle (radians) and longitude of line of nodes OMEGA
    (radians, measured anti-clockwise from North on the sky).

    Sky axes: due West = x, due North = y, towards Earth = z.
    """
    xd,yd,zd = d2s_axes(iangle, OMEGA)
    return (x*xd.x + y*yd.x, x*xd.y + y*yd.y)

def d2s_axes(iangle, OMEGA):
    """
    Returns x,y,z basis vectors defined by the disc in terms of the sky axes.
    x of the disc is in the plane of the sky along the line of nodes. If
    OMEGA=0 it points North. The z-axis is defined to be perpendicular to the
    the disc and towards the observer for iangle < 90. The y-axis is defined
    so that a right-handed Cartesian triad emerges.  iangle is the angle
    between the disc's z axis and the line of sight.

    The sky axes are defined by due West = x, due North = y, towards Earth (line of sight)
    = z.

    This returns a tuple of Vec3 objects (xd,yd,zd)
    """
    coso, sino = math.cos(OMEGA), math.sin(OMEGA)
    cosi, sini = math.cos(iangle), math.sin(iangle)

    xd = Vec3(-sino, coso, 0)
    yd = Vec3(-cosi*coso, -cosi*sino, -sini)
    zd = Vec3(-sini*coso, -sini*sino,  cosi)

    return (xd,yd,zd)

def lc2(x, y, f, tf, rd,
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
        fdisc = dflux2(x, y, f, tf, rd, r1, x1s[i], y1s[i], z1s[i], r2, x2s[i], y2s[i], z2s[i])

        # compute what we see from the two stars. we ignore the
        # possibility of the disc occulting the stars.
        f1,f2 = ring.flux2(r, rings, fluxes, tflux, (x1s[i],x2s[i]), (y1s[i],y2s[i]), (z1s[i],z2s[i]))

        # Multiply in surface brightnesses to get final number.
        lc[i] = s1*f1 + s2*f2 + fdisc

    return lc
