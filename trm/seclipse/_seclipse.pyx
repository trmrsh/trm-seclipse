import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport acos, sqrt, atan2, M_PI

DTYPE = np.float64
ITYPE = np.int
ctypedef np.float64_t DTYPE_t
ctypedef np.int_t ITYPE_t

@cython.cdivision(True)
cdef visible(double rring, double r, double x, double y):
    """Computes the visible range of a circle (ring) of radius 'rring', when
    obscured by an opaque circle of radius 'r' offset from the centre of the
    ring by (x,y).

    Arguments::

      rring : (float)
        radius of ring being occulted

      r    : (float)
        radius of occulter

      x    : (float)
        x offset of centre of occulter from centre of ring

      y    : (float)
        y offset of centre of occulter from centre of ring


    Returns (lo,hi) representing the start and end of the visible range as a
    fraction of the circle measured counter-clockwise from the x-axis.

    If the ring is completely visible, then hi >= lo + 1.  If the ring is
    completely obscured, then hi <= lo. Otherwise, 0 < lo < 1, hi > lo, hi <
    lo + 1.

    """

    cdef double psq, p, lo, hi, theta, cosp, phi

    # compute "impact parameter" p.
    psq = x**2+y**2
    p = sqrt(psq)

    if p >= rring + r or rring >= p + r:
        # ring fully visible (obscuring circle either fully outside or fully
        # inside it)
        lo,hi = 0.,3.

    elif p + rring <= r:
        # ring fully obscured
        lo,hi = 0.,-1.

    else:
        # partial case. polar angle to centre of obscuring circle from centre
        # of ring
        theta = atan2(y, x)

        # cosine of half angle subtended by obscuring circle
        cosp = (rring**2+psq-r**2)/(2.*rring*p)

        phi = acos(cosp)
        lo = (theta+phi)/(2.*M_PI)
        hi = (theta-phi)/(2.*M_PI) + 1
        if lo < 0:
            lo += 1
            hi += 1

    return (lo,hi)

cdef overlap(double l1, double h1, double l2, double h2):
    """Routine to combine two visibility ranges l1 to h1, l2 to h2, representing
    contiguous, potentially overlapping, visible fractions of a circle,
    returning either a single region or two disjoint regions of
    visibility. (See 'visible' for possible values of l1, h1 etc.)

    Arguments::

       l1 : (float)
          lower limit of visible region 1

       h1 : (float)
          upper limit of visible region 1

       l2 : (float)
          lower limit of visible region 2

       h2 : (float)
          upper limit of visible region 2

    Returns (l1,h1,l2,h2)

    This routine can be used to build up the visible regions allowed by the
    eclipse of multiple stars. Its outcomes are:

      1) Zero regions: h1 <= l1.

      2) One region: this will be represented by l1 to h1 with l1 < h1 <= l1 +
      1 and h2 <= l2

      3) Two disjoint regions: l1 to h1, l2 to h2 with l1 < h1 <= l1 + 1 and
      l2 < h2 <= l2 + 1

    The returned ranges never extend more than 1 cycle. Input ranges must also
    not exceed 1. l1, l2 must lie from 0 to 1 on input. Not checked.

    """
    cdef double sl2, sh2

    if h1 <= l1 or h2 <= l2:
        # nothing visible
        l1, h1, l2, h2 = 0., -1., 0., -1.

    elif l1 <= l2 and h1 >= h2:
        # 1 spans 2 so 2 defines the visible section.
        l1, h1 = l2, h2
        l2, h2 = 0., -1.

    elif l2 <= l1 and h2 >= h1:
        # 2 spans 1 so 1 defines the visible section
        l2, h2 = 0., -1.

    elif l2 >= h1:
        # 2 lies beyond 1 around the circle
        if h2 > l1 + 1:
            # 2 wraps round and overlaps start of 1.
            h1 = min(h2-1,h1)
            l2, h2 = 0., -1.
        else:
            # no overlap
            l1, h1, l2, h2 = 0., -1., 0., -1.

    elif l1 >= h2:
        # 1 lies beyond 2 around the circle
        if h1 > l2 + 1:
            # 1 wraps round and overlaps 2
            l1 = l2
            h1 = min(h1-1,h2)
            l2, h2 = 0., -1.
        else:
            # no overlap
            l1, h1, l2, h2 = 0., -1., 0., -1.

    elif l1 <= l2 and l2 <= h1:
        # Ranges overlap, 1 left of 2.
        sl2, sh2 = l2, h2
        if h2 > l1 + 1.:
            # wraps creating two regions
            l2 = l1
            h2 = h2-1.
        else:
            # fails to wrap leaving one region
            l2, h2 = 0., -1.
        l1 = sl2
        h1 = min(h1, sh2)

    elif l2 <= l1 and l1 <= h2:
        # Ranges overlap, 2 left of 1.
        sh2 = h2
        if h1 > l2 + 1.:
            h2 = h1-1.
        else:
            l2, h2 = 0., -1.
        h1 = min(h1, sh2)

    else:
        raise Exception('l1, h1, l2, h2 = {:f}, {:f}, {:f}, {:f}'.format(l1,h1,l2,h2))

    return (l1,h1,l2,h2)

cdef double comb2(double l1, double h1, double l2, double h2):
    """Computes total fraction of a circle visible given two possibly overlapping
    visibility ranges. See 'visible' for an explanation of the possible values
    of l1, h1 etc. For instance l1=0, h1=0.5 & l2=0.4, h2=0.9 leaves only the
    portion 0.4 to 0.5 visible, and would return a value of 0.1.  Needed in 3
    sphere computation.

    Arguments::

      l1 : (float)
         start of range visible behind star 1

      h1 : (float)
         end of range visible behind star 1

      l2 : (float)
         start of range visible behind star 2

      h2 : (float)
         end of range visible behind star 2

    Returns single number representing fraction of circle visible.

    """
    cdef double total

    l1, h1, l2, h2 = overlap(l1, h1, l2, h2)

    total = 0.
    if h1 > l1: total += min(h1-l1, 1.)
    if h2 > l2: total += min(h2-l2, 1.)

    return total

cdef double comb3(double l1, double h1, double l2, double h2, double l3, double h3):
    """Computes total fraction visible given three, possibly overlapping,
    visibility ranges. Needed in 4 sphere computation.

    Arguments::

      l1 : (float)
         start of range visible behind star 1

      h1 : (float)
         end of range visible behind star 1

      l2 : (float)
         start of range visible behind star 2

      h2 : (float)
         end of range visible behind star 2

      l3 : (float)
         start of range visible behind star 3

      h3 : (float)
         end of range visible behind star 3

    Returns single number representing fraction of circle visible.

    """
    cdef double total, l2t, h2t

    total = 0.

    # combine first two regions into 0, 1 or 2 non-overlapping regions
    l1, h1, l2, h2 = overlap(l1, h1, l2, h2)

    if h1 > l1:
        # work out part(s) visible through new region 1 and old 3
        l1, h1, l2t, h2t = overlap(l1, h1, l3, h3)
        if h1 > l1: total += min(h1-l1, 1)
        if h2t > l2t: total += min(h2t-l2t, 1)

    if h2 > l2:
        # work out part(s) visible through new region 2 and old 3
        l1, h1, l2, h2 = overlap(l2, h2, l3, h3)
        if h1 > l1: total += min(h1-l1, 1)
        if h2 > l2: total += min(h2-l2, 1)

    return total

cdef double fvis1(double rring, double r, double x, double y):
    """Computes fraction of a ring visible behind one occulting disc.

    Arguments::

      rring : (float)
        radius of ring being occulted

      r    : (float)
        radius of occulter

      x    : (float)
        x offset of centre of occulter from centre of ring

      y    : (float)
        y offset of centre of occulter from centre of ring

    Returns a single number, the fraction of the ring that is visible.
     """

    # compute visibility ranges
    cdef double l, h
    l, h = visible(rring, r, x, y)
    return max(0., min(1., h-l))

cdef double fvis2(double rring, double r1, double x1, double y1, double r2, double x2, double y2):
    """Computes fraction of a ring visible behind two occulting discs.

    Arguments::

      rring : (float)
        radius of ring being occulted

      r1   : (float)
        radius of occulter 1

      x1   : (float)
        x offset of centre of occulter 1 from centre of ring

      y1   : (float)
        y offset of centre of occulter 1 from centre of ring

      r2   : (float)
        radius of occulter 2

      x2   : (float)
        x offset of centre of occulter 2 from centre of ring

      y2   : (float)
        y offset of centre of occulter 2 from centre of ring

    Returns a single number, the fraction of the ring that is visible.
    """

    cdef double l1, h1, l2, h2

    # compute visibility ranges
    l1, h1 = visible(rring, r1, x1, y1)
    l2, h2 = visible(rring, r2, x2, y2)

    return comb2(l1, h1, l2, h2)


cdef double fvis3(
    double rring,
    double r1, double x1, double y1,
    double r2, double x2, double y2,
    double r3, double x3, double y3):
    """Routine to compute fraction of a ring of radius rring
    visible behind three discs.

    Arguments::

      rring : (float)
        radius of ring being occulted

      r1   : (float)
        radius of occulter 1

      x1   : (float)
        x offset of centre of occulter 1 from centre of ring

      y1   : (float)
        y offset of centre of occulter 1 from centre of ring

      r2   : (float)
        radius of occulter 2

      x2   : (float)
        x offset of centre of occulter 2 from centre of ring

      y2   : (float)
        y offset of centre of occulter 2 from centre of ring

      r3   : (float)
        radius of occulter 3

      x3   : (float)
        x offset of centre of occulter 3 from centre of ring

      y3   : (float)
        y offset of centre of occulter 3 from centre of ring

    Returns a single number, the fraction of the ring that is visible.

    """

    cdef double l1, h1, l2, h2, l3, h3

    # compute visibility ranges
    l1, h1 = visible(rring, r1, x1, y1)
    l2, h2 = visible(rring, r2, x2, y2)
    l3, h3 = visible(rring, r3, x3, y3)

    return comb3(l1, h1, l2, h2, l3, h3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ring1(np.ndarray[DTYPE_t, ndim=1, mode='c'] rings,
                  np.ndarray[DTYPE_t, ndim=1, mode='c'] fluxes,
                  double tflux, double r, double x, double y):
    """Computes the flux from a star potentially occulted by 1 other.

    Arguments::

       rings  : (array)
          radii of regularly-spaced rings covering face of star being occulted.

       fluxes : (array)
          maximum flux contributions from each ring

       tflux  : (float)
          total flux (sum over fluxes, a time saver)

       r      : (float)
          radius of occulting star

       x      : (float)
          x offset of occulting star from centre of rings

       y      : (float)
          y offset of occulting star from centre of rings

    Returns total visible flux.

    See also ring2, ring3 and ring4

    """
    cdef unsigned int i, n = len(rings)
    cdef double p, rring, flux, sum

    # try to speed things a bit with an early bail out. This will often be
    # helpful because most of the time, the stars will not align.
    p = sqrt(x**2+y**2)
    if p >= rings[n-1]+r:
        return tflux

    # ok, need to calculate
    sum = 0.
    for i in range(n):
        sum += fluxes[i]*fvis1(rings[i], r, x, y)

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ring2(np.ndarray[DTYPE_t, ndim=1] rings, np.ndarray[DTYPE_t, ndim=1] fluxes,
                  double tflux, double r1, double x1, double y1, double r2, double x2, double y2):
    """Computes the flux from a star potentially occulted by 2 others.

    Arguments::

       rings  : (array)
          radii of regularly-spaced rings covering face of star being occulted.

       fluxes : (array)
          maximum flux contributions from each ring

       tflux  : (float)
          total flux (sum over fluxes, a time saver)

       r1     : (float)
          radius of occulting star 1

       x1     : (float)
          x offset of occulting star 1 from centre of rings

       y1     : (float)
          y offset of occulting star 1 from centre of rings

       r2     : (float)
          radius of occulting star 2

       x2     : (float)
          x offset of occulting star 2 from centre of rings

       y2     : (float)
          y offset of occulting star 2 from centre of rings

    Returns total visible flux.

    See also ring1, ring3 and ring4

    """

    cdef unsigned int i, n = len(rings)
    cdef double p1, p2, sum, rring, flux, rout = rings[n-1]

    # fast bail out
    p1 = sqrt(x1**2 + y1**2)
    p2 = sqrt(x2**2 + y2**2)
    if p1 >= rout+r1 and p2 >= rout+r2:
        return tflux

    # get calculating
    sum = 0.
    for i in range(n):
        # partial by both 1 and 2
        sum += fluxes[i]*fvis2(rings[i], r1, x1, y1, r2, x2, y2)

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ring3(np.ndarray[DTYPE_t, ndim=1] rings, np.ndarray[DTYPE_t, ndim=1] fluxes,
                  double tflux,
                  double r1, double x1, double y1,
                  double r2, double x2, double y2,
                  double r3, double x3, double y3):
    """Computes the flux from a star potentially occulted by 3 others.

    Arguments::

       rings  : (array)
          radii of regularly-spaced rings covering face of star being occulted.

       fluxes : (array)
          maximum flux contributions from each ring

       tflux  : (float)
          total flux (sum over fluxes, a time saver)

       r1     : (float)
          radius of occulting star 1

       x1     : (float)
          x offset of occulting star 1 from centre of rings

       y1     : (float)
          y offset of occulting star 1 from centre of rings

       r2     : (float)
          radius of occulting star 2

       x2     : (float)
          x offset of occulting star 2 from centre of rings

       y2     : (float)
          y offset of occulting star 2 from centre of rings

       r3     : (float)
          radius of occulting star 3

       x3     : (float)
          x offset of occulting star 3 from centre of rings

       y3     : (float)
          y offset of occulting star 3 from centre of rings

    Returns total visible flux.

    See also ring1, ring2 and ring4

    """

    cdef unsigned int i, n = len(rings)
    cdef double p1, p2, p3, sum, rring, flux, rout = rings[n-1]

    # fast bail out
    p1 = sqrt(x1**2+y1**2)
    p2 = sqrt(x2**2+y2**2)
    p3 = sqrt(x3**2+y3**2)
    if p1 >= rout+r1 and p2 >= rout+r2 and p3 >= rout+r3:
        return tflux

    # get calculating
    sum = 0.
    for i in range(n):
        # partial obscuration by all 3.
        sum += fluxes[i]*fvis3(rings[i], r1, x1, y1, r2, x2, y2, r3, x3, y3)

    return sum

def flux2(r, rings, fluxes, tflux, x, y, z):
    """Computes flux from each of three spheres, accounting for their mutual
    eclipses. All arguments here are array-like with 3 elements each, covering
    the 3 spheres.

    Arguments::

      r       : (list/tuple of floats)
         radii of spheres [2 values]

      rings   : (tuple of arrays)
         arrays of radii of annuli covering the faces of each sphere [2
         arrays]

      fluxes  : (tuple of arrays)
         arrays of flux contributions from the annuli covering each
         sphere. This allows limb darkening to be accounted for. [2 arrays]

      tflux   : (tuple of arrays)
         total flux contributions from each sphere, i.e. the totals of the
         'fluxes' arrays (time saver) [2 values]

      x       : (tuple of floats)
         x ordinates of centres of positions of the centre of each sphere [2
         values]

      y       : (tuple of floats)
         y ordinates of centres of positions of the centre of each sphere [2
         values]

      z       : (list/tuple of floats)
         z ordinates of centres of positions of the centre of each sphere. The
         z-axis must point towards Earth (crucial for ordering the spheres) [2
         values]

    Returns (f1,f2) where f1 and f2 are the total visible fluxes from
    each sphere.

    Use 'rfinit' to compute the radii, fluxes and tflux for each sphere in
    order to generate the inputs to this routine.

    """

    cdef unsigned int i1, i2
    cdef double f1, f2

    # Determine the order of the stars, furthest --> nearest from Earth.
    i1, i2 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star i1 is behind star i2
    f1 = ring1(rings[i1], fluxes[i1], tflux[i1], r[i2], x[i2]-x[i1], y[i2]-y[i1])

    # star i2 is unobscured
    f2 = tflux[i2]

    # unscramble the sort order
    fs = [0,0]
    fs[i1], fs[i2] = f1, f2
    return tuple(fs)

def flux3(r, rings, fluxes, tflux, x, y, z):
    """Computes flux from each of three spheres, accounting for their mutual
    eclipses. All arguments here are array-like with 3 elements each, covering
    the 3 spheres.

    Arguments::

      r       : (list/tuple of floats)
         radii of spheres [3 values]

      rings   : (list/tuple of arrays)
         arrays of radii of annuli covering the faces of each sphere [3
         arrays]

      fluxes  : (list/tuple of arrays)
         arrays of flux contributions from the annuli covering each
         sphere. This allows limb darkening to be accounted for. [3 arrays]

      tflux : (list/tuple of arrays)
         total flux contributions from each sphere, i.e. the totals of the
         'fluxes' arrays (time saver) [3 values]

      x       : (list/tuple of floats)
         x ordinates of centres of positions of the centre of each sphere [3
         values]

      y       : (list/tuple of floats)
         y ordinates of centres of positions of the centre of each sphere [3
         values]

      z       : (list/tuple of floats)
         z ordinates of centres of positions of the centre of each sphere. The
         z-axis must point towards Earth (crucial for ordering the spheres) [3
         values]

    Returns (f1,f2,f3) where f1, f2, and f3 are the total visible fluxes fro
    each sphere.

    Use 'rfinit' to compute the radii, fluxes and tflux for each sphere in
    order to generate the inputs to this routine.

    """

    cdef unsigned int i1, i2, i3
    cdef double f1, f2, f3

    # Determine the order of the stars, furthest --> nearest from Earth.
    i1, i2, i3 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star i1 is behind stars i2 and i3
    f1 = ring2(rings[i1], fluxes[i1], tflux[i1],
               r[i2], x[i2]-x[i1], y[i2]-y[i1],
               r[i3], x[i3]-x[i1], y[i3]-y[i1])

    # star i2 is behind star i3
    f2 = ring1(rings[i2], fluxes[i2], tflux[i2],
               r[i3], x[i3]-x[i2], y[i3]-y[i2])

    # star i3 is unobscured
    f3 = tflux[i3]

    # unscramble the sort order
    fs = [0,0,0]
    fs[i1], fs[i2], fs[i3] = f1, f2, f3
    return tuple(fs)

def lc3(r, rings, fluxes, tflux, s1, s2, s3, p1s, p2s, p3s):
    """"Wrapper around 'flux3' to run over lots of positions specified in the 
    p1s, etc tuples of x,y,z positions for each star.
    """

    cdef unsigned int i, n
    cdef double f1, f2, f3
    cdef np.ndarray[DTYPE_t, ndim=1] x1s,y1s,z1s,x2s,y2s,z2s,x3s,y3s,z3s,lc

    # unpack the tuples
    (x1s,y1s,z1s),(x2s,y2s,z2s),(x3s,y3s,z3s) = p1s,p2s,p3s

    n = len(x1s)
    lc = np.empty(n)

    for i in range(n):
        # Construct x,y,z position tuples
        x = (x1s[i], x2s[i], x3s[i])
        y = (y1s[i], y2s[i], y3s[i])
        z = (z1s[i], z2s[i], z3s[i])

        # Call flux3
        f1,f2,f3 = flux3(r, rings, fluxes, tflux, x, y, z)

        # Multiply in surface brightnesses to get final number.
        lc[i] = s1*f1 + s2*f2 + s3*f3

    return lc

def flux4(r, rings, fluxes, tflux, x, y, z):
    """Computes flux from each of four spheres, accounting for their mutual
    eclipses. All arguments here are array-like with 4 elements each, covering
    the 4 spheres.

    Arguments::

      r       : (list/tuple of floats)
         radii of spheres [4 values]

      rings   : (list/tuple of arrays)
         arrays of radii of annuli covering the faces of each sphere [4
         arrays]

      fluxes  : (list/tuple of arrays)
         arrays of flux contributions from the annuli covering each
         sphere. This allows limb darkening to be accounted for. [4 arrays]

      tflux : (list/tuple of arrays)
         total flux contributions from each sphere, i.e. the totals of the
         'fluxes' arrays (time saver) [4 values]

      x       : (list/tuple of floats)
         x ordinates of centres of positions of the centre of each sphere [4
         values]

      y       : (list/tuple of floats)
         y ordinates of centres of positions of the centre of each sphere [4
         values]

      z       : (list/tuple of floats)
         z ordinates of centres of positions of the centre of each sphere. The
         z-axis must point towards Earth (crucial for ordering the spheres) [4
         values]


    Returns (f1,f2,f3) where f1, f2, f3 and f4 are the total visible fluxes
    fro each sphere.

    Use 'rfinit' to compute the radii, fluxes and tflux for each sphere in
    order to generate the inputs to this routine.

    """

    cdef unsigned int i1, i2, i3, i4
    cdef double f1, f2, f3, f4

    # Determine the order of the stars, furthest --> nearest from Earth.
    i1, i2, i3, i4 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star index i1 is behind stars i2, i3 and i4
    f1 = ring3(rings[i1], fluxes[i1], tflux[i1],
               r[i2], x[i2]-x[i1], y[i2]-y[i1],
               r[i3], x[i3]-x[i1], y[i3]-y[i1],
               r[i4], x[i4]-x[i1], y[i4]-y[i1])

    # star i2 is behind stars i3 and i4
    f2 = ring2(rings[i2], fluxes[i2], tflux[i2],
               r[i3], x[i3]-x[i2], y[i3]-y[i2],
               r[i4], x[i4]-x[i2], y[i4]-y[i2])

    # star i3 is behind star i4
    f3 = ring1(rings[i3], fluxes[i3], tflux[i3],
               r[i4], x[i4]-x[i3], y[i4]-y[i3])

    # star i4 is unobscured
    f4 = tflux[i4]

    # unscramble the sort order
    fs = [0,0,0,0]
    fs[i1], fs[i2], fs[i3], fs[i4] = f1, f2, f3, f4
    return tuple(fs)

def lc4(r, rings, fluxes, tflux, s1, s2, s3, s4, p1s, p2s, p3s, p4s):
    """"Wrapper around 'flux4' to run over lots of positions specified in the
    p1s, etc tuples of x,y,z positions for each star.
    """

    cdef unsigned int i, n
    cdef double f1, f2, f3, f4
    cdef np.ndarray[DTYPE_t, ndim=1] x1s,y1s,z1s,x2s,y2s,z2s,x3s,y3s,z3s,x4s,y4s,z4s,lc

    # unpack the tuples
    (x1s,y1s,z1s),(x2s,y2s,z2s),(x3s,y3s,z3s),(x4s,y4s,z4s) = p1s,p2s,p3s,p4s

    n = len(x1s)
    lc = np.empty(n)

    for i in range(n):
        # Construct x,y,z position tuples
        x = (x1s[i], x2s[i], x3s[i], x4s[i])
        y = (y1s[i], y2s[i], y3s[i], y4s[i])
        z = (z1s[i], z2s[i], z3s[i], z4s[i])

        # Call flux4
        f1,f2,f3,f4 = flux4(r, rings, fluxes, tflux, x, y, z)

        # Multiply in surface brightnesses to get final number.
        lc[i] = s1*f1 + s2*f2 + s3*f3 + s4*f4

    return lc

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def expand(np.ndarray[DTYPE_t, ndim=1] ts, np.ndarray[DTYPE_t, ndim=1] tes,
           np.ndarray[ITYPE_t, ndim=1] nds):
    """Expands a set of times, exposures and integer sub-division factors into
    an array of times that can be used to smear each individual
    exposure. i.e. the output is a larger array in which each original time
    ts[i] becomes a set of nds[i] times equally spaced around the input time
    and spanning a length tes[i].
    """

    cdef unsigned int i, n, j
    cdef np.ndarray[DTYPE_t, ndim=1] tout = np.empty(nds.sum(), dtype=DTYPE)

    # counter
    n = 0
    for i in range(len(ts)):
        if nds[i] > 1:
            for j in range(nds[i]):
                tout[n+j] = ts[i]+tes[i]*(j-(nds[i]-1)/2)/(nds[i]-1)
        else:
            tout[n] = ts[i]
        n += nds[i]

    return tout

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compress(np.ndarray[DTYPE_t, ndim=1] fs, np.ndarray[ITYPE_t, ndim=1] nds):
    """Compresses a set of values (typically fluxes) assumed evaluated at a
    series of expanded times as returned by 'expand' with integer sub-division
    factors 'nds'. The compression is carried out through trapezoidal averaging
    of the values contributing to each output value.
    """

    cdef unsigned int i, j, n, nout=len(nds)
    cdef double sum
    cdef np.ndarray[DTYPE_t, ndim=1] fout = np.empty(nout, dtype=DTYPE)

    # counter
    n = 0
    for i in range(nout):
        if nds[i] > 1:
            sum = (fs[n]+fs[n+nds[i]-1])/2
            n += 1
            for j in range(nds[i]-2):
                sum += fs[n+j]
            n += nds[i]-1
            fout[i] = sum / (nds[i]-1)
        else:
            fout[i] = fs[n]
            n += 1

    return fout


@cython.wraparound(False)
@cython.boundscheck(False)
def dflux2(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] f,
           double tf, double rd,
           double r1, double x1, double y1, double z1, double r2, double x2, double y2, double z2):
    """Computes the flux from a disc potentially occulted by 2 spheres. The x and y arrays
    should be computed from "disc.rfinit" followed by "disc.project".

    Arguments::

      x   : (array)
         X-ordinates of disc elements on plane of sky

      y   : (array)
         Y-ordinates of disc elements on plane of sky

      f   : (array)
         Flux contributions of disc elements

      tf  : (float)
         Total flux from disc.

      rd  : (float)
         Radius of disc (used as a crude time saver)

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

    cdef unsigned int i, n = len(f)
    cdef double r1sq, r2sq, xsq, flux = 0.

    # The second condition in the next two lines is a minimal condition for a
    # sphere to occult the disc. i.e. if it is not satisfied, there will be no
    # eclipse, but if it is satisfied, there may or may not be an eclipse.  It
    # is added to save time because quite often this routine is likely to be
    # called with the objects quite far apart on the sky
    occ1 = z1 > 0 and x1**2+y1**2 < (rd+r1)**2
    occ2 = z2 > 0 and x2**2+y2**2 < (rd+r2)**2

    r1sq = r1**2
    r2sq = r2**2

    if occ1 and occ2:
        for i in range(n):
            if (x[i]-x1)**2+(y[i]-y1)**2 > r1sq and (x[i]-x2)**2+(y[i]-y2)**2 > r2sq:
                flux += f[i]
    elif occ1:
        for i in range(n):
            if (x[i]-x1)**2+(y[i]-y1)**2 > r1sq:
                flux += f[i]
    elif occ2:
        for i in range(n):
            if (x[i]-x2)**2+(y[i]-y2)**2 > r2sq:
                flux += f[i]
    else:
        # no occultation of disc
        flux = tf
    return flux
