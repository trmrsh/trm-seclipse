import sys
import math
import numpy as np
import numexpr as ne
import trm.subs as subs

def circle(phase, iangle, s1, s2, r1, r2, limb1, limb2, n1, n2, b1, b2):
    """Computes light curve for a circular orbit given phases and inclination
    angle. Works by splitting up visible faces of stars into a series of
    annuli, basically a wrapper around "flux". Look at flux for remaining
    parameters.

    Arguments::

      phase : (array)
          orbital phases

      iangle : (float)
          inclination angle [degrees]

      s1 : (float)
          surface brightness star 1

      s2 : (float)
          surface brightness star 1

      r1 : (float)
          radius star 1, scaled by the separation

      r2 : (float)
          radius star 2, scaled by the separation

      limb1 : (Limb)
          limb darkening star 1

      limb2 : (Limb)
          limb darkening star 2

      n1 : (int)
          number of annuli star 1

      n2 : (int)
          number of annuli star 2

      b1 : (float)
          beaming factor star 1 (all constants combined
          so just multiplies sine, > 0)

      b2 : (float)
          beaming factor star 2 (> 0)

    """
    sini = math.sin(math.radians(iangle))
    prad = (2.*math.pi)*phase
    cosp = np.cos(prad)
    sinp = np.sin(prad)
    p    = np.sign(cosp)*np.sqrt(1-(sini*cosp)**2)
    return flux(p, s1*(1+b1*sinp), s2*(1-b2*sinp), r1, r2, limb1, limb2, n1, n2)

def flux(p, s1, s2, r1, r2, limb1, limb2, n1, n2):
    """
    Computes total flux from two stars of radius r1 and r2
    separated on the sky by p (an array). p > 0 means star 1
    is further away than star 2; p < 0 means star 1 is closer
    than star 2.

    Arguments::

      p : (array)
          impact parameters between two stars

      s1 : (array)
          surface brightnesses, star 1 (array to allow beaming)

      s2 : (array)
          surface brightnesses, star 2 (array to allow beaming)

      r1 : (float)
          radius star 1

      r2 : (float)
          radius star 2

      limb1 : (Limb)
          limb darkening star 1

      limb2 : (Limb)
          limb darkening star 2

      n1 : (int)
          number of annuli star 1

      n2 : (int)
          number of annuli star 2
    """

    # note that since p, s1 and s2 are arrays, looping is used
    # here to reduce memory requirements inside integ. also
    f      = np.empty_like(p)
    mflux1 = integ(r1,limb1,n1,r2,2.*(r1+r2))
    mflux2 = integ(r2,limb2,n2,r1,2.*(r1+r2))

    for i,pim in enumerate(p):
        if abs(pim) >= r1+r2:
            f[i] = mflux1*s1[i] + mflux2*s2[i]
        elif pim >= 0.:
            f[i] = integ(r1,limb1,n1,r2,pim)*s1[i] + mflux2*s2[i]
        else:
            f[i] = mflux1*s1[i] + integ(r2,limb2,n2,r1,pim)*s2[i]

    return f

def fluxes(s1, s2, r1, r2, limb1, limb2, n1, n2):
    """Computes flux from each star separately, useful in determining the flux
    ratio.

    Arguments::

      s1 : (array)
          surface brightnesses, star 1

      s2 : (array)
          surface brightnesses, star 2

      r1 : (float)
          radius star 1

      r2 : (float)
          radius star 2

      limb1 : (Limb)
          limb darkening star 1

      limb2 : (Limb)
          limb darkening star 2

      n1 : (int)
          number of annuli star 1

      n2 : (int)
          number of annuli star 2

    """

    return (
        s1*integ(r1,limb1,n1,r2,2.*(r1+r2)),
        s2*integ(r2,limb2,n2,r1,2.*(r1+r2))
    )


def integ(r1, limb1, n1, r2, p):
    """Computes flux from a star of unit central surface brightnes, radius r1,
    limb darkening limb1, being occulted by a star of radius r2. p is the
    projected distance between the centres of the stars. n1 is the number of
    annuli to use for the integration.

    Arguments::

      r1 : (float)
         radius of occulted star

      limb1 : (Limb)
         limb darkening, function of mu.

      n1 : (int)
         number of annuli

      r2 : (float)
         radius of occulter

      p : (float)
         impact parameter

    """

    p = abs(p)
    dsq = p**2 - r2**2

    # range of radii where occultation happens
    rmin = p - r2
    rmax = p + r2

    # generate radii of annuli and trapezoidal weights
    # effectively radii start at 0 end at r1, except we
    # drop the zero one from the start, leaving only the
    # outermost one with a half weight.
    r = np.linspace( r1/n1, r1, n1)
    w = np.ones_like(r)
    w[-1] = 0.5

    # keep only those annuli not fully obscured
    cansee = r > -rmin
    r = r[cansee]
    w = w[cansee]

    # compute mu values
    mu = np.sqrt(1-(r/r1)**2)

    # add any fully exposed annuli
    full = (r <= rmin) | (r >= rmax)
    if full.any():
        wf = w[full]
        rf = r[full]
        lf = limb1(mu[full])
        sum = 2.*math.pi*(ne.evaluate('wf*rf*lf').sum())
    else:
        sum = 0.

    # add any partially exposed annuli
    pi = math.pi
    if not full.all():
        part = ~full
        wp = w[part]
        rp = r[part]
        lp  = limb1(mu[part])
        cost = ne.evaluate('(rp**2+dsq)/(2.*rp*p)')
        theta = np.arccos(cost)
        sum += 2.*(ne.evaluate('(pi-theta)*wp*rp*lp').sum())

    # scale by radial step
    return sum*r1/n1

def visible(ring, r, x, y):
    """Computes the visible range of a circle (ring) of radius 'ring', when
    obscured by an opaque circle of radius 'r' offset from the centre of the
    ring by (x,y).

    Arguments::

      ring : (float)
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

    # compute "impact parameter" p.
    psq = x**2+y**2
    p = math.sqrt(psq)

    if p >= ring+r:
        # ring fully visible
        lo,hi = 0,3
    elif p + ring <= r:
        # ring fully obscured
        lo,hi = 0,-1
    else:
        # partial case. polar angle to centre of obscuring circle from centre
        # of ring
        theta = math.atan2(y, x)

        # cosine of half angle subtended by obscuring circle
        cosp = (ring**2+psq-r**2)/(2.*ring*p)
        phi = math.acos(cosp)
        lo = (theta+phi)/(2.*math.pi)
        hi = (theta-phi)/(2.*math.pi) + 1
        if lo < 0:
            lo += 1
            hi += 1

    return (lo,hi)

def overlap(l1, h1, l2, h2):
    """Routine to combine two visibility ranges l1 to h1, l2 to h2, representing
    contiguous, potentially overlapping, visible fractions of a circle,
    returning a single region or two disjoint regions of visibility. (See
    'visible' for possible values of l1, h1 etc.)

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

    if l1 <= l2 and h1 >= h2:
        # 1 spans 2 so 2 defines the visible section.
        l1 = l2
        h1 = h2
        l2, h2 = 0, -1

    elif l2 <= l1 and h2 >= h1:
        # 2 spans 1 so 1 defines the visible section
        l2, h2 = 0, -1

    elif l2 >= h1:
        # 2 lies beyond 1 around the circle
         if h2 > l1 + 1:
            # 2 wraps round and overlaps 1.
            h1 = min(h2-1,h1)
            l2, h2 = 0, -1
        else:
            # no overlap
            l1, h1, l2, h2 = 0, -1, 0, -1

    elif l1 >= h2:
        # 1 lies beyond 2 around the circle

        if h1 > l2 + 1:
            # 1 wraps round and overlaps 2
            l1 = l2
            h1 = std::min(h1-1,h2)
        else:
            # no overlap
            l1, h1, l2, h2 = 0, -1, 0, -1

    elif l1 <= l2 and l2 <= h1:
        # Ranges overlap, 1 left of 2.
        sl2 = l2
        if h2 > l1 + 1:
            # wraps creating two regions
            l2 = l1
            h2 = h2-1
        else:
            # fails to wrap leaving one region
            l2, h2 = 0, -1
        l1 = sl2

    elif l2 <= l1 and l1 <= h2:
        # Ranges overlap, 2 left of 1.
        sh2 = h2
        if h1 > l2 + 1:
            h2 = h1-1
        else:
            l2, h2 = 0, -1
        h1 = sh2

    else:
        raise Exception('l1, h1, l2, h2 = {:f}, {:f}, {:f}, {:f}'.format(l1,h1,l2,h2))

    return (l1,h1,l2,h2)

def comb2(l1, h1, l2, h2):
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

    l1, h1, l2, h2 = overlap(l1, h1, l2, h2)

    total = 0.
    if h1 > l1: total += h1-l1
    if h2 > l2: total += h2-l2

    return total

def comb3(l1, h1, l2, h2, l3, h3):
    """Computes total fraction visible given three, possibly
    overlapping, visibility ranges. Needed in 4 sphere computation.

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

    # sort so l1 < l2 < l3
    ranges = [(l1,h1),(l2,h2),(l3,h3)]
    ranges.sort()
    (l1,h1),(l2,h2),(l3,h3) = ranges

    total = 0.

    # combine first two regions
    l1, h2, l2, h2 = overlap(l1, h1, l2, h2)

    if h1 > l1:
        # work out part(s) visible through new region 1 and old 3
        l2t, h2t = l3, h3
        l1, h1, l2t, h2t = overlap(l1, h1, l2t, h2t)
        if h1 > l1: total += h1-l1
        if h2t > l2t: total += h2t-l2t

    if h2 > l2:
        # work out part(s) visible through new region 2 and old 3
        l2t, h2t = l3, h3
        l2, h2, l2t, h2t = overlap(l2, h2, l2t, h2t)
        if h2 > l2: total += h2-l2
        if h2t > l2t: total += h2t-l2t

    return total

def fvis1(ring, r, x, y):
    """Computes fraction of a ring visible behind one occulting disc.

    Arguments::

      ring : (float)
        radius of ring being occulted

      r    : (float)
        radius of occulter

      x    : (float)
        x offset of centre of occulter from centre of ring

      y    : (float)
        y offset of centre of occulter from centre of ring

    Returns a single number, the fraction of the ring that is visible.
     """

    # compute "impact parameter" p.
    psq = x**2+y**2
    p = math.sqrt(psq)

    if p >= ring+r:
        vis = 1
    elif p + ring <= r:
        vis = 0
    else:
        # cosine of half angle subtended by obscuring circle
        cosp = (ring**2+psq-r**2)/(2.*ring*p)
        vis = 1-math.acos(cosp)/math.pi
    return vis


def fvis2(ring, x1, y1, r1, x2, y2, r2):
    """Computes fraction of a ring visible behind two occulting discs.

    Arguments::

      ring : (float)
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
    # compute visibility ranges
    l1, h1 = visible(ring, r1, x1, y1)
    l2, h2 = visible(ring, r2, x2, y2)

    return comb2(l1, h1, l2, h2)


def fvis3(ring, r1, x1, y1, r2, x2, y2, r3, x3, y3):
    """Routine to compute fraction of a ring of radius ring
    visible behind three discs.

    Arguments::

      ring : (float)
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

    # compute visibility ranges
    l1, h1 = visible(ring, r1, x1, y1)
    l2, h2 = visible(ring, r2, x2, y2)
    l3, h3 = visible(ring, r3, x3, y3)

    return comb3(l1, h1, l2, h2, l3, h3)

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

def ring1(rings, fluxes, tflux, r, x, y):
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

    # try to speed things a bit with an early bail out. This will often be
    # helpful because most of the time, the stars will not align.
    p = math.sqrt(x**2+y**2)
    if p >= rings[-1]+r:
        return tflux

    # do it the hard way
    dsq  = p**2 - r**2
    rmin = p - r
    rmax = p + r

    sum = 0.
    for ring, flux in zip(rings, fluxes):
        if ring <= -rmin:
            # completely obscured, add nothing
            continue

        elif ring <= rmin or ring >= rmax:
            # completely unobscured, add the lot
            sum += flux

        else:
            # partial obscured
            sum += flux*fvis1(rings, r, x, y)

    return sum

def ring2(rings, fluxes, tflux, r1, x1, y1, r2, x2, y2):
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

    # fast bail out
    p1 = math.sqrt(x1**2+y1**2)
    p2 = math.sqrt(x2**2+y2**2)
    if p1 >= rings[-1]+r1 and p2 >= rings[-1]+r2:
        return tflux

    # get calculating
    dsq1  = p1**2 - r1**2
    dsq2  = p2**2 - r2**2
    rmin1 = p1 - r1
    rmax1 = p1 + r1
    rmin2 = p2 - r2
    rmax2 = p2 + r2

    sum = 0.
    for ring, flux in zip(rings, fluxes):

        if ring <= -rmin1 or ring <= -rmin2:
            # completely obscured
            continue

        elif (ring <= rmin1 or ring >= rmax1) and \
                (ring <= rmin2 or ring >= rmax2):
            # no obscuration
            sum += flux

        elif ring <= rmin1 or ring >= rmax1:
            # partial obscuration by 2
            sum += flux*fvis1(ring, r2, x2, y2)

        elif ring <= rmin2 or ring >= rmax2:
            # partial obscuration by 1
            sum += flux*fvis1(ring, r1, x1, y2)

        else:
            # partial by both 1 and 2
            sum += flux*fvis2(ring, r1, x1, y1, r2, x2, y2)

    return sum

def ring3(rings, fluxes, tflux, r1, x1, y1, r2, x2, y2, r3, x3, y3):
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

    # fast bail out
    p1 = math.sqrt(x1**2+y1**2)
    p2 = math.sqrt(x2**2+y2**2)
    p3 = math.sqrt(x3**2+y3**2)
    if p1 >= rings[-1]+r1 and p2 >= rings[-1]+r2 and p3 >= rings[-1]+r3:
        return tflux

    # get calculating
    dsq1 = p1**2 - r1**2
    dsq2 = p2**2 - r2**2
    dsq3 = p3**2 - r3**2
    rmin1 = p1 - r1
    rmax1 = p1 + r1
    rmin2 = p2 - r2
    rmax2 = p2 + r2
    rmin3 = p3 - r3
    rmax3 = p3 + r3

    sum = 0.
    for ring, flux in zip(rings, fluxes):

        if ring <= -rmin1 or ring <= -rmin2 or ring <= -rmin3:
            # complete obscuration
            continue

        elif (ring <= rmin1 or ring >= rmax1) and \
                (ring <= rmin2 or ring >= rmax2) and \
                (ring <= rmin3 or ring >= rmax3):
            # no obscuration at all
            sum += flux

        elif (ring <= rmin1 or rings >= rmax1) and \
                (ring <= rmin2 or ring >= rmax2):
            # partial obscuration by 3 only
            sum += flux*fvis1(ring, r3, x3, y3)

        elif (ring <= rmin1 or ring >= rmax1) and \
                (ring <= rmin3 || ring >= rmax3):
            # partial obscuration by 2 only
            sum += flux*fvis1(ring, r2, x2, y2)

        elif (ring <= rmin2 or ring >= rmax2) and \
                (ring <= rmin3 or rings[i] >= rmax3):
            # partial obscuration by 1 only
            sum += flux*fvis1(ring, r1, x1, y2)

        elif ring <= rmin1 or ring >= rmax1:
            # partial by 2 and 3
            sum += flux*fvis2(ring, r2, x2, y2, r3, x3, y3)

        elif ring <= rmin2 or ring >= rmax2:
            # partial by 1 and 3
            sum += flux*fvis2(ring, r1, x1, y1, r3, x3, y3)

        elif ring <= rmin3 or ring >= rmax3:
            # partial by 2 and 1
            sum += flux*fvis2(ring, r2, x2, y2, r1, x1, y1)

        else:
            # partial obscuration by all 3.
            sum += flux*fvis3(ring, r1, x1, y1, r2, x2, y2, r3, x3, y3)

    return sum

def flux3(r1, r2, r3, limb1, limb2, limb3, n1, n2, n3, p1, p2, p3):
    """Computes flux from each of three limb darkened spheres, each of unit
    central surface brightness, accounting for their mutual eclipses.

    Arguments::

      r1 : (float)
         radius of sphere 1

      r2 : (float)
         radius of sphere 2

      r3 : (float)
         radius of sphere 3

      limb1 : (Limb)
         limb darkening of sphere 1

      limb2 : (Limb)
         limb darkening of sphere 2

      limb3 : (Limb)
         limb darkening of sphere 3

      n1 : (int)
         number of annuli covering face of sphere 1

      n2 : (int)
         number of annuli covering face of sphere 2

      n3 : (int)
         number of annuli covering face of sphere 3

      p1 : (trm.subs.Vec3)
         x,y,z position of centre of star 1. The z-axis
         must point towards Earth.

      p2 : (trm.subs.Vec3)
         x,y,z position of centre of star 2

      p3 : (trm.subs.Vec3)
         x,y,z position of centre of star 3

    Returns (f1,f2,f3) where::

      f1 : (float)
        flux from sphere 1

      f3 : (float)
        flux from sphere 2

      f3 : (float)
        flux from sphere 3

    The function is designed to be called repeatedly with differing positions
    but everything else (r1, limb1, n1 etc) staying the same.  Under these
    circumstances it stores many of the variables required for repeated
    evaluation.

    """

    # This is likely to be called multiple times with identical
    # values of r1, r2, r3, limb1, limb2, limb3, n1, n2, n3 so
    # we can save some time by doing some calculations once and
    # storing the values of r1 etc to test whether any have changed.
    if not hasattr(flux3,'rings') or (r1,r2,r3) != flux3.r or \
            (n1,n2,n3) != flux3.n or (limb1,limb2,limb3) != flux3.limb:

        rings1, fluxes1, tflux1 = rfinit(r1, limb1, n1)
        rings2, fluxes2, tflux2 = rfinit(r2, limb2, n2)
        rings3, fluxes3, tflux3 = rfinit(r3, limb3, n3)
        flux3.rings = (rings1,rings2,rings3)
        flux3.fluxes = (fluxes1,fluxes2,fluxes3)
        flux3.tflux = (tflux1,tflux2,tflux3)
        flux3.r = (r1,r2,r3)
        flux3.n = (n1,n2,n3)
        flux3.limb = (limb1,limb2,limb3)

    # Determine the order of the stars, furthest --> nearest from Earth.
    x, y, z = (p1.x, p2.x, p3.x), (p1.y, p2.y, p3.y), (p1.z, p2.z, p3.z)
    i1, i2, i3 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star index i1 is behind stars i2 and i3
    f1 = ring2(flux3.rings[i1], flux3.fluxes[i1], flux3.tflux[i1],
               r[i2], x[i2]-x[i1], y[i2]-y[i1],
               r[i3], x[i3]-x[i1], y[i3]-y[i1])

    # star i2 is behind star i3
    f2 = ring1(flux3.rings[i2], flux3.fluxes[i2], flux3.tflux[i2],
               r[i3], x[i3]-x[i2], y[i3]-y[i2])

    # star i3 is unobscured
    f3 = flux3.tflux[i3]

    # ensure the fluxes are returned in the correct order
    fs = [0,0,0]
    fs[i1], fs[i2], fs[i3] = f1, f2, f3
    return tuple(fs)

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

    if not hasattr(flux4,'rings') or r != flux4.r or n != flux4.n or \
       limb != flux4.limb:

        rings1, fluxes1, tflux1 = rfinit(r1, limb1, n1)
        rings2, fluxes2, tflux2 = rfinit(r2, limb2, n2)
        rings3, fluxes3, tflux3 = rfinit(r3, limb3, n3)
        rings4, fluxes4, tflux4 = rfinit(r4, limb4, n4)
        flux4.rings = (rings1,rings2,rings3,rings4)
        flux4.fluxes = (fluxes1,fluxes2,fluxes3,fluxes4)
        flux4.tflux = (tflux1,tflux2,tflux3,tflux4)
        flux4.r, flux.n, flux4.limb = r, n, limb

    # Determine the order of the stars, furthest --> nearest from Earth.
    x, y, z = (p1.x, p2.x, p3.x, p4.x), (p1.y, p2.y, p3.y, p4.y), \
              (p1.z, p2.z, p3.z, p4.z)
    i1, i2, i3, i4 = np.array(z).argsort()

    # ok, now evaluate the fluxes from each star

    # star index i1 is behind stars i2, i3 and i4
    f1 = ring3(flux4.rings[i1], flux4.fluxes[i1], flux4.tflux[i1],
               r[i2], x[i2]-x[i1], y[i2]-y[i1],
               r[i3], x[i3]-x[i1], y[i3]-y[i1],
               r[i4], x[i4]-x[i1], y[i4]-y[i1])

    # star i2 is behind stars i3 and i4
    f2 = ring2(flux4.rings[i2], flux4.fluxes[i2], flux4.tflux[i2],
               r[i3], x[i3]-x[i2], y[i3]-y[i2],
               r[i4], x[i4]-x[i2], y[i4]-y[i2])

    # star i3 is behind star i4
    f3 = ring1(flux4.rings[i3], flux4.fluxes[i3], flux4.tflux[i3],
               r[i4], x[i4]-x[i3], y[i4]-y[i3])

    # star i4 is unobscured
    f4 = flux4.tflux[i4]

    # ensure the fluxes are returned in the correct order
    fs = [0,0,0,0]
    fs[i1], fs[i2], fs[i3], fs[i4] = f1, f2, f3, f4
    return tuple(fs)

class Limb (object):
    """
    Limb darkening object
    """

    POLY   = 1
    CLARET = 2

    def __init__(self, ltype, a1, a2=0, a3=0, a4=0):
        """
        Initialiser / constructor of Limb objects.

        Arguments::

          ltype : (int)
              either Limb.POLY or Limb.CLARET to indicate type of limb
              darkening

          a1    : (float)
              first coeffient

          a2    : (float)
              second coeffient

          a3    : (float)
              third coeffient

          a4    : (float)
              fourth coeffient
        """
        if ltype != Limb.POLY and ltype != Limb.CLARET:
            raise Exception('Invalid limb darkening type.')
        self.ltype = ltype
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def __repr__(self):
        rep = 'Limb(type=' + repr(self.ltype) + ', a1=' + repr(self.a1) + \
            ', a2=' + repr(self.a2) + ', a3=' + repr(self.a3) + \
            ', a4=' + repr(self.a4) + ')'
        return rep

    def __call__(self, mu):
        ommu = 1.-mu
        im   = 1.
        a1, a2, a3, a4 =self.a1, self.a2, self.a3, self.a4
        if self.ltype == Limb.POLY:
            im -= ne.evaluate('ommu*(a1+ommu*(a2+ommu*(a3+ommu*a4)))')
        elif self.ltype == Limb.CLARET:
            im -= a1+a2+a3+a4
            msq = np.sqrt(mu)
            im += ne.evaluate('msq*(a1+msq*(a2+msq*(a3+msq*a4)))')
        return im
