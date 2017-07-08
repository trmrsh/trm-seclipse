"""
Implements an alternative but slower way to calculate sphere eclipses
in which the visible faces of the stars are spliy into small elements.
"""

import sys
import math
import numpy as np
import numexpr as ne
import trm.subs as subs

def rfinit(r, limb, n):
    """Initialises the grid elements and flux contributions using a series of n
    annuli covering the visible face of a star of unit central surface
    brightness.

    Arguments::

      r : (float)
         stellar radius

      limb : (Limb)
         limb darkening

      n : (int)
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
    ngrid = 0
    for radius in radii:
        ngrid += max(8, int(math.pi*radius/halfdr))
    x = np.empty(ngrid)
    y = np.empty(ngrid)
    f = np.empty(ngrid)

    ngrid = 0
    for radius in radii:
        # compute number of thetas to make elements roughly square
        ntheta = max(8, int(math.pi*radius/halfdr))

        # compute x, y [on sky] coordinates
        thetas = np.linspace(0,2.*math.pi*(1-1/ntheta),ntheta)
        xa = radius*np.cos(thetas)
        ya = radius*np.sin(thetas)

        # now the fluxes
        mu = np.sqrt(1-(radius/r)**2)
        fa = (radius*(2.*math.pi/ntheta)*(r/n)*limb(mu))*np.ones(ntheta)

        # store values just calculated
        x[ngrid:ngrid+ntheta] = xa
        y[ngrid:ngrid+ntheta] = ya
        f[ngrid:ngrid+ntheta] = fa

        # update offset pointer
        ngrid += ntheta

    return (x, y, f, f.sum())

def flux4(r1, r2, r3, r4, limb1, limb2, limb3, limb4,
          n1, n2, n3, n4, p1, p2, p3, p4):
    """Computes flux from each of four limb darkened spheres, each of unit
    central surface brightness, accounting for their mutual eclipses.

    Arguments::

      r1 : (float)
         radius of sphere 1

      r2 : (float)
         radius of sphere 2

      r3 : (float)
         radius of sphere 3

      r4 : (float)
         radius of sphere 4

      limb1 : (Limb)
         limb darkening of sphere 1

      limb2 : (Limb)
         limb darkening of sphere 2

      limb3 : (Limb)
         limb darkening of sphere 3

      limb4 : (Limb)
         limb darkening of sphere 4

      n1 : (int)
         number of annuli covering face of sphere 1

      n2 : (int)
         number of annuli covering face of sphere 2

      n3 : (int)
         number of annuli covering face of sphere 3

      n4 : (int)
         number of annuli covering face of sphere 4

      p1 : (trm.subs.Vec3)
         x,y,z position of centre of star 1. The z-axis
         must point towards Earth.

      p2 : (trm.subs.Vec3)
         x,y,z position of centre of star 2

      p3 : (trm.subs.Vec3)
         x,y,z position of centre of star 3

      p4 : (trm.subs.Vec3)
         x,y,z position of centre of star 4

    Returns (f1,f2,f3,f4) where::

      f1 : (float)
        flux from sphere 1

      f3 : (float)
        flux from sphere 2

      f3 : (float)
        flux from sphere 3

      f4 : (float)
        flux from sphere 4

    The function is designed to be called repeatedly with differing positions
    but everything else (r1, limb1, n1 etc) staying the same.  Under these
    circumstances it stores many of the variables required for repeated
    evaluation.

    """

    # This is likely to be called multiple times with identical values of r1,
    # r2, r3, r4, limb1, limb2, limb3, limb4, n1, n2, n3, n4 so we can save
    # some time by doing some calculations once and storing the values of r1
    # etc to test whether any have changed.

    # create tuples
    r, n, limb = (r1,r2,r3,r4), (n1,n2,n3,n4), (limb1,limb2,limb3,limb4)

    if not hasattr(flux4,'fluxes') or r != flux4.r or n != flux4.n or \
       limb != flux4.limb:

        x1, y1, fluxes1, tflux1 = rfinit(r1, limb1, n1)
        x2, y2, fluxes2, tflux2 = rfinit(r2, limb2, n2)
        x3, y3, fluxes3, tflux3 = rfinit(r3, limb3, n3)
        x4, y4, fluxes4, tflux4 = rfinit(r4, limb4, n4)

        flux4.x = (x1,x2,x3,x4)
        flux4.y = (y1,y2,y3,y4)
        flux4.fluxes = (fluxes1,fluxes2,fluxes3,fluxes4)
        flux4.tflux = (tflux1,tflux2,tflux3,tflux4)
        flux4.r, flux.n, flux4.limb = r, n, limb

    # Determine the order of the stars, furthest --> nearest from Earth.
    x, y, z = (p1.x, p2.x, p3.x, p4.x), (p1.y, p2.y, p3.y, p4.y), \
              (p1.z, p2.z, p3.z, p4.z)
    i1, i2, i3, i4 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star index i1 is behind stars i2, i3 and i4. Compute boolean arrays
    # to show which of its elements can be seen
    vis = ((fluxes.x[i1]-x[i2])**2 + (fluxes.y[i1]-y[i2])**2 > r[i2]**2) & \
          ((fluxes.x[i1]-x[i3])**2 + (fluxes.y[i1]-y[i3])**2 > r[i3]**2) & \
          ((fluxes.x[i1]-x[i4])**2 + (fluxes.y[i1]-y[i4])**2 > r[i4]**2)

    f1 = flux4.fluxes[i1][vis].sum()

    # star i2 is behind stars i3 and i4
    vis = ((fluxes.x[i2]-x[i3])**2 + (fluxes.y[i2]-y[i3])**2 > r[i3]**2) & \
          ((fluxes.x[i2]-x[i4])**2 + (fluxes.y[i2]-y[i4])**2 > r[i4]**2)

    f2 = flux4.fluxes[i2][vis].sum()

    # star i3 is behind star i4
    vis = ((fluxes.x[i3]-x[i4])**2 + (fluxes.y[i3]-y[i4])**2 > r[i4]**2)

    f3 = flux4.fluxes[i3][vis].sum()

    # star i4 is unobscured
    f4 = flux4.tflux[i4]

    # ensure the fluxes are returned in the correct order
    fs = [0,0,0,0]
    fs[i1], fs[i2], fs[i3], fs[i4] = f1, f2, f3, f4
    return tuple(fs)
