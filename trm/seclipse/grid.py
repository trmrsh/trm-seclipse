"""Calculates the eclipse of limb darkened spheres by splitting each sphere's
visible face into a grid of elements arranged in a series of equal increment
annuli each split into equal angular increments so that they are approximately
square. This should be compared to the method implemented in the 'rings' set
of routines.  'grid' is fundamentally 2D but makes better use of numpy's
vectorised speed up.

"""

import sys
import math
import numpy as np
import numexpr as ne

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

def flux4(r, rings, fluxes, tflux, x, y, z):
    """Computes flux from each of four spheres, accounting for their mutual
    eclipses. All arguments here are array-like with 4 elements each, covering
    the 4 spheres.

    Arguments::

      r       : (tuple of floats)
         radii of spheres [4 values]

      rings   : (tuple of arrays)
         arrays of the radii of annuli covering the faces of each star [4 arrays]

      fluxes  : (tuple of arrays)
         arrays of the fluxes of annuli covering the faces of each star [4 arrays]

      tfluxes  : (tuple of floats)
         total flux contributions from each sphere [4 values]

      x       : (tuple of floats)
         x ordinates of centres of positions of the centre of each sphere [4
         values]

      y       : (tuple of floats)
         y ordinates of centres of positions of the centre of each sphere [4
         values]

      z       : (tuple of floats)
         z ordinates of centres of positions of the centre of each sphere. The
         z-axis must point towards Earth (crucial for ordering the spheres) [4
         values]

    The arguments are deisgned so that for multiple phases of a given model,
    r, xs, ys, fluxes, tflux will not change, while xs, ys and zs probably
    will.

    Returns (f1,f2,f3) where f1, f2, f3 and f4 are the total visible fluxes
    fro each sphere.

    Use 'rfinit' to compute the xs, ys, fluxes and tflux for each sphere in
    order to generate the inputs to this routine.

    """

    # Determine the order of the stars, furthest --> nearest from Earth.
    i1, i2, i3, i4 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star index i1 is behind stars i2, i3 and i4. Compute boolean arrays to
    # show which of its elements can be seen
    vis = \
        ((xs[i1] + (x[i1]-x[i2]))**2 + (ys[i1] + (y[i1]-y[i2]))**2 > r[i2]**2) & \
        ((xs[i1] + (x[i1]-x[i3]))**2 + (ys[i1] + (y[i1]-y[i3]))**2 > r[i3]**2) & \
        ((xs[i1] + (x[i1]-x[i4]))**2 + (ys[i1] + (y[i1]-y[i4]))**2 > r[i4]**2)

    f1 = flux4.fluxes[i1][vis].sum()

    # star i2 is behind stars i3 and i4
    vis = \
        ((xs[i2] + (x[i2]-x[i3]))**2 + (ys[i2] + (y[i2]-y[i3]))**2 > r[i3]**2) & \
        ((xs[i2] + (x[i2]-x[i4]))**2 + (ys[i2] + (y[i2]-y[i4]))**2 > r[i4]**2)

    f2 = flux4.fluxes[i2][vis].sum()

    # star i3 is behind star i4
    vis = ((xs[i3] + (x[i3]-x[i4]))**2 + (ys[i3] + (y[i3]-y[i4]))**2 > r[i4]**2)

    f3 = flux4.fluxes[i3][vis].sum()

    # star i4 is unobscured
    f4 = flux4.tflux[i4]

    # ensure the fluxes are returned in the correct order
    fs = [0,0,0,0]
    fs[i1], fs[i2], fs[i3], fs[i4] = f1, f2, f3, f4
    return tuple(fs)
