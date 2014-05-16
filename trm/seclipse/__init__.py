#!/usr/bin/env python

"""
a module to compute the eclipse of limb-darkened spheres. It has routines
for two and three spheres.
"""

from __future__ import division

import sys
import collections
import math as m
import numpy as np
import numexpr as ne
import trm.subs as subs

sys.path.append('.')

def circle(phase, iangle, s1, s2, r1, r2, limb1, limb2, n1, n2, b1, b2):
    """
    Computes light curve for circular orbit given phases and inclination
    angle. Look at flux for remaining parameters.

      phase : array
          orbital phases

      iangle : float
          inclination angle

      s1 : float
          surface brightness star 1

      s2 : float
          surface brightness star 1

      r1 : float
          scaled radius star 1

      r2 : float
          scaled radius star 2

      limb1 : float
          limb darkening star 1

      limb2 : float
          limb darkening star 2

      n1 : int
          number of annuli star 1

      n2 : int
          number of annuli star 2

      b1 : float
          beaming factor star 1 (all constants combined
          so just multiplies sine, > 0)

      b2 : float
          beaming factor star 2 (> 0)
    """
    sini = m.sin(m.radians(iangle))
    prad = (2.*m.pi)*phase
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

      p : array
          impact parameters between two stars

      s1 : array
          surface brightnesses, star 1

      s2 : array
          surface brightnesses, star 2

      r1 : float
          radius star 1

      r2 : float
          radius star 2

      limb1 : float
          limb darkening star 1

      limb2 : float
          limb darkening star 2

      n1 : int
          number of annuli star 1

      n2 : int
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
    """
    Computes flux from each star separately, useful in determining the flux
    ratio

      s1 : array
          surface brightnesses, star 1

      s2 : array
          surface brightnesses, star 2

      r1 : float
          radius star 1

      r2 : float
          radius star 2

      limb1 : float
          limb darkening star 1

      limb2 : float
          limb darkening star 2

      n1 : int
          number of annuli star 1

      n2 : int
          number of annuli star 2
    """

    return s1*integ(r1,limb1,n1,r2,2.*(r1+r2)), \
        s2*integ(r2,limb2,n2,r1,2.*(r1+r2))


def integ(r1, limb1, n1, r2, p):
    """
    Computes flux from a star of unit central surface brightnes, radius r1,
    limb darkening limb1, being occulted by a star of radius r2. p is the
    projected distance between the centres of the stars. n1 is the number of
    annuli to use for the integration.

    r1 : float
       radius of occulted star

    limb1 : float
       limb darkening, function of mu.

    n1 : int
       number of annuli

    r2 : float
       radius of occulter

    p : float
       impact parameter

    """

    p    = abs(p)
    dsq  = p**2 - r2**2

    # range of radii where occultation happens
    rmin = p - r2
    rmax = p + r2

    # generate radii of annuli and trapezoidal weights
    # effectively radii start at 0 end at r1, except we
    # drop the zero one from the start, leaving only the
    # outermost one with a half weight.
    r     = np.linspace( r1/n1, r1, n1)
    w     = np.ones_like(r)
    w[-1] = 0.5

    # keep only those annuli not fully obscured
    cansee = r > -rmin
    r = r[cansee]
    w = w[cansee]

    # compute mu values
    mu   = np.sqrt(1-(r/r1)**2)

    # add any fully exposed annuli
    full = (r <= rmin) | (r >= rmax)
    if full.any():
        wf   = w[full]
        rf   = r[full]
        lf   = limb1(mu[full])
        sum  = 2.*m.pi*(ne.evaluate('wf*rf*lf').sum())
    else:
        sum = 0.

    # add any partially exposed annuli
    pi = m.pi
    if not full.all():
        part  = ~full
        wp    = w[part]
        rp    = r[part]
        lp    = limb1(mu[part])
        cost  = ne.evaluate('(rp**2+dsq)/(2.*rp*p)')
        theta = np.arccos(cost)
        sum  += 2.*(ne.evaluate('(pi-theta)*wp*rp*lp').sum())

    # scale by radial step
    return sum*r1/n1

def ring1(rings, r, x, y):
    """
    Computes the fraction of a series of rings visible behind a star.
    This is a basic step in computing mutual eclipses between two spheres
    where the rings represent annuli over the face of the most
    distant star.

    Arguments::

      rings : array
          set of radii of concentric rings

      r : float
          radius of star

      x : float
          separation in x on sky of centre of star from centre of rings

      y : float
          separation in y on sky of centre of star from centre of rings

    Returns (full, total, vis) where::

      full : bool
          True if the rings are not obscured at all

      total : bool
          True if the rings are totally obscured.

      vis : array
          array of visibilities, ranging from 0 to 1 for each ring. If
          full=True, these will all be 1, if total=True these will all
          be 0.

    See also ring2
    """

    twopi = 2.*m.pi

    p = np.sqrt(x**2+y**2)

    # now calculate overlap (if any) with the rings
    dsq  = p**2 - r**2

    # range of radii where occultation happens
    rmin = p - r
    rmax = p + r

    # work out visibility status of rings. the ones that need work are those
    # which are neither totally obscured or fully visible, so we make sure to
    # do these and no others.
    none = rings <= -rmin
    full = (rings <= rmin) | (rings >= rmax)
    part = ~none & ~full

    # fully visible by default
    vis = np.ones_like(rings)

    # set obscured rings to 0
    vis[none] = 0.

    # compute angular extent of partially exposed parts
    if part.any():
        rp = rings[part]
        # cosp = ne.evaluate('(rp1**2+dsq1)/(2.*rp1*p1)')
        # calculate phi, the half-angle subtended by the obscured
        # section as seen from the centre of the ring.
        cosp = (rp**2+dsq)/(2.*rp*p)
        phi = np.arccos(cosp)
        vis[part] = 1-phi/m.pi

    return (full.all(), none.all(), vis)

def ring2(rings, r1, x1, y1, r2, x2, y2):
    """
    Computes the fraction of a series of rings visible behind two stars
    This is a basic step in computing mutual eclipses between three
    spheres where the rings represent annuli over the face of the most
    distant star.

    Arguments::

      rings : array
          set of radii of concentric rings

      r1 : float
          radius of star 1

      x1 : float
          separation in x on sky of centre of star 1 from centre of rings

      y1 : float
          separation in y on sky of centre of star 1 from centre of rings

      r2 : float
          radius of star 2

      x2 : float
          separation in x on sky of centre of star 2 from centre of rings

      y2 : float
          separation in y on sky of centre of star 2 from centre of rings

    Returns (full, total, vis) where::

      full : bool
          True if the rings are not obscured at all

      total : bool
          True if the rings are totally obscured.

      vis : array
          array of visibilities, ranging from 0 to 1 for each ring. If
          full=True, these will all be 1, if total=True these will all
          be 0.

    See also ring1
    """

    twopi = 2.*m.pi

    # compute impact parameters of each star relative to the rings
    p1 = np.sqrt(x1**2 + y1**2)
    p2 = np.sqrt(x2**2 + y2**2)

    # now go through each star calculating overlap (if any)
    # with the rings

    # star1
    dsq1  = p1**2 - r1**2

    # range of radii where occultation happens
    rmin1 = p1 - r1
    rmax1 = p1 + r1

    # star2
    dsq2  = p2**2 - r2**2

    # range of radii where occultation happens
    rmin2 = p2 - r2
    rmax2 = p2 + r2

    # work out visibility status of rings. the ones that need work are those
    # which are neither totally obscured or fully visible, so we make sure to
    # do these and no others.
    none  = (rings <= -rmin1) | (rings <= -rmin2)

    # back to star 1
    full1 = (rings <= rmin1) | (rings >= rmax1)
    part1 = ~none & ~full1

    # compute angular extent of partially exposed parts
    if part1.any():
        theta1 = m.atan2(y1, x1)

        rp1 = rings[part1]
        # cosp = ne.evaluate('(rp1**2+dsq1)/(2.*rp1*p1)')
        # calculate phi, the half-angle subtended by the obscured
        # section as seen from the centre of the ring.
        cosp1 = (rp1**2+dsq1)/(2.*rp1*p1)
        phi1 = np.arccos(cosp1)

        # lo1 --> hi1 extent of visible section. Put into array
        # of same length as rings for ease of use later. Map
        # lower limit into range 0 to 1
        lo  = (theta1 + phi1)/twopi
        hi  = (theta1 - phi1)/twopi + 1
        neg = lo < 0.
        lo[neg] += 1
        hi[neg] += 1

        lo1 = np.empty_like(rings)
        hi1 = np.empty_like(rings)
        lo1[part1] = lo
        hi1[part1] = hi

    # and star 2 again
    full2 = (rings <= rmin2) | (rings >= rmax2)
    part2 = ~none & ~full2

    # compute angular extent of partially exposed parts
    if part2.any():
        theta2 = m.atan2(y2, x2)
        rp2 = rings[part2]

        # cosp = ne.evaluate('(rp1**2+dsq1)/(2.*rp1*p1)')
        # calculate phi, the half-angle subtended by the obscured
        # section as seen from the centre of the ring.
        cosp2 = (rp2**2+dsq2)/(2.*rp2*p2)
        phi2 = np.arccos(cosp2)

        # lo2 --> hi2 extent of visible section. Put into array
        # of same length as rings for ease of use later. Map
        # lower limit into range 0 to 1
        lo  = (theta2 + phi2)/twopi
        hi  = (theta2 - phi2)/twopi + 1
        neg = lo < 0.
        lo[neg] += 1
        hi[neg] += 1

        lo2 = np.empty_like(rings)
        hi2 = np.empty_like(rings)
        lo2[part2] = lo
        hi2[part2] = hi

    # Compute fractional visibility of each ring (0 to 1)

    # fully visible by default
    vis = np.ones_like(rings)

    # set obscured rings to 0
    vis[none] = 0.

    # now the partial ones which are only partial for one
    # out of the two stars
    only1 = part1 & full2
    if only1.any():
        vis[only1] = hi1[only1]-lo1[only1]

    only2 = part2 & full1
    if only2.any():
        vis[only2] = hi2[only2]-lo2[only2]


    # finally those which are partial for both stars
    # resort to a loop here ...
    both = part1 & part2
    if both.any():
        vpart = vis[both]
        n = 0
        for l1,h1,l2,h2 in zip(lo1[both],hi1[both],lo2[both],hi2[both]):

            # 8 separate cases. I wonder if this is all of them or
            # whether it can be shortened?
            if l1 <= l2 and h1 >= h2:
                # range 1 covers range 2
                vpart[n] = h1-l1

            elif l2 <= l1 and h2 >= h1:
                # range 2 covers range 1
                vpart[n] = h2-l2

            elif l1 <= l2 and l2 <= h1:
                # range 2 overlaps and extends range 1
                # NB we know that hi2 > hi1 here because
                # otherwise test 1 would have worked
                vpart[n] = h2-l1

            elif l2 <= l1 and l1 <= h2:
                # range 1 overlaps and extends range 2
                # NB we know that hi1 > hi2 here because
                # otherwise test 2 would have worked
                vpart[n] = h1-l2

            elif l2 <= l1 and h2 >= h1:
                # range 2 covers range 1
                vpart[n] = h2-l2

            elif h1 <= l2 and h2 >= l1+1:
                # previous tests should have picked up
                # all direct overlaps but we need to remember
                # cyclic nature of ranges. Here we test for range
                # 1 lying to the left of range 2, but range 2
                # overlaping range1 on its (range1's) next cycle
                vpart[n] = h1+1-l2

            elif h2 <= l1 and h1 >= l2+1:
                # as previous but with 1 and 2 swapped
                vpart[n] = h2+1-l1

            else:
                # ranges disjoint.
                vpart[n] = h1-l1 + h2-l2

            n += 1

    full  = (full1 & full2).all()
    total = none.all()

    return (full, total, vis)

def fout(r, limb, n):
    """
    Computes flux from an isolated, spherical star of unit central surface
    brightness by summation over a series of concentric annuli covering its
    visible face. Useful as a time saver to establish the flux out of eclipse
    (hence "fout"). Returns arrays of ring radii and contributions to the
    flux ring by ring. The sum over the latter is the total flux.
    to the sum from each ring.

    Arguments::

      r : float
         stellar radius

      limb : Limb
         limb darkening

      n : int
         number of annuli.

    Returns (rings, fluxes) where::

      rings : array
         ring radii to use when computing obscuration

      fluxes: array
         unocculted contributions of each ring to the total flux
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

    # returns the results
    return (rings, (2.*m.pi*r/n)*(w*rings*lmb))

def flux3(r1, r2, r3, limb1, limb2, limb3, n1, n2, n3, p1, p2, p3):
    """
    Computes flux from each of three limb darkened spheres, each of unit
    central surface brightness, accounting for their mutual eclipses
    given::

      r1 : float
         radius of sphere 1

      r2 : float
         radius of sphere 2

      r3 : float
         radius of sphere 3

      limb1 : Limb
         limb darkening of sphere 1

      limb2 : Limb
         limb darkening of sphere 2

      limb3 : Limb
         limb darkening of sphere 3

      n1 : int
         number of annuli covering face of sphere 1

      n2 : int
         number of annuli covering face of sphere 2

      n3 : int
         number of annuli covering face of sphere 3

      p1 : trm.subs.Vec3
         x,y,z position of centre of star 1. The z-axis
         must point towards Earth.

      p2 : trm.subs.Vec3
         x,y,z position of centre of star 2

      p3 : trm.subs.Vec3
         x,y,z position of centre of star 3


    Returns (f1,f2,f3) where::

      f1 : float
        flux from sphere 1

      f3 : float
        flux from sphere 2

      f3 : float
        flux from sphere 3

    The function is designed to be called repeatedly with differing
    positions but everything else (r1, limb1, n1 etc) staying the same.
    Under these circumstances it stores many of the variables required
    for repeated evaluation.
    """

    # speed ups for repeat calls
    if not hasattr(flux3,'rings1') or r1 != flux3.r1 or \
            n1 != flux3.n1 or limb1 is not flux3.limb1:
        flux3.r1 = r1
        flux3.limb1 = limb1
        flux3.n1 = n1
        flux3.rings1, flux3.fluxes1 = fout(r1, limb1, n1)
        flux3.f1 = flux3.fluxes1.sum()

    if not hasattr(flux3,'fout2') or r2 != flux3.r2 or \
            n2 != flux3.n2 or limb2 is not flux3.limb2:
        flux3.r2 = r2
        flux3.limb2 = limb2
        flux3.n2 = n2
        flux3.rings2, flux3.fluxes2 = fout(r2, limb2, n2)
        flux3.f2 = flux3.fluxes2.sum()

    if not hasattr(flux3,'fout3') or r3 != flux3.r3 or \
            n3 != flux3.n3 or limb3 is not flux3.limb3:
        flux3.r3 = r3
        flux3.limb3 = limb3
        flux3.n3 = n3
        flux3.rings3, flux3.fluxes3 = fout(r3, limb3, n3)
        flux3.f3 = flux3.fluxes3.sum()

    # ok now do the work which is largely book-keeping
    if p1.z <= p2.z and p1.z <= p3.z:
        # stars 2 & 3 in front of star 1
        full1, total1, vis1 = ring2(flux3.rings1,
                                    r2, p2.x-p1.x, p2.y-p1.y,
                                    r3, p3.x-p1.x, p3.y-p1.y)
        if p2.z < p3.z:
            # star 3 in front of star 2
            full2, total2, vis2 = ring1(flux3.rings2,
                                        r3, p3.x-p2.x, p3.y-p2.y)
            full3 = True
        else:
            # star 2 in front of star 3
            full3, total3, vis3 = ring1(flux3.rings3,
                                        r2, p2.x-p3.x, p2.y-p3.y)
            full2 = True

    elif p2.z <= p1.z and p2.z <= p3.z:
        # stars 1 & 3 in front of star 2
        full2, total2, vis2 = ring2(flux3.rings2,
                                    r1, p1.x-p2.x, p1.y-p2.y,
                                    r3, p3.x-p2.x, p3.y-p2.y)
        if p1.z < p3.z:
            # star 3 in front of star 1
            full1, total1, vis1 = ring1(flux3.rings1,
                                        r3, p3.x-p1.x, p3.y-p1.y)
            full3 = True
        else:
            # star 1 in front of star 3
            full3, total3, vis3 = ring1(flux3.rings3,
                                        r1, p1.x-p3.x, p1.y-p3.y)
            full1 = True

    elif p3.z <= p1.z and p3.z <= p2.z:
        # stars 1 & 2 in front of star 3
        full3, total3, vis3 = ring2(flux3.rings3,
                                    r1, p1.x-p3.x, p1.y-p3.y,
                                    r3, p2.x-p3.x, p2.y-p3.y)
        if p1.z < p2.z:
            # star 2 in front of star 1
            full1, total1, vis1 = ring1(flux3.rings1,
                                        r2, p2.x-p1.x, p2.y-p1.y)
            full2 = True
        else:
            # star 1 in front of star 2
            full2, total2, vis2 = ring1(flux3.rings2,
                                        r1, p1.x-p2.x, p1.y-p2.y)
            full1 = True

    # finally compute output fluxes
    if full1:
        f1 = flux3.f1
    elif total1:
        f1 = 0.
    else:
        f1 = (vis1*flux3.fluxes1).sum()

    if full2:
        f2 = flux3.f2
    elif total2:
        f2 = 0.
    else:
        f2 = (vis2*flux3.fluxes2).sum()

    if full3:
        f3 = flux3.f3
    elif total3:
        f3 = 0.
    else:
        f3 = (vis3*flux3.fluxes3).sum()

    if f1 > 1.e10 or f1 < -1.e10:
        print 'f1 =',f1
        print 'vis1 =',vis1
    return (f1,f2,f3)

class Limb (object):
    """
    Limb darkening object
    """

    POLY   = 1
    CLARET = 2

    def __init__(self, type, a1, a2=0, a3=0, a4=0):
        if type == Limb.POLY:
            self.type = Limb.POLY
        elif type == Limb.CLARET:
            self.type = Limb.CLARET
        else:
            raise Exception('Invalid limb darkening type.')
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def __repr__(self):
        rep = 'Limb(type=' + repr(self.type) + ', a1=' + repr(self.a1) + \
            ', a2=' + repr(self.a2) + ', a3=' + repr(self.a3) + \
            ', a4=' + repr(self.a4) + ')'
        return rep

    def __call__(self, mu):
        ommu = 1.-mu
        im   = 1.
        a1, a2, a3, a4 =self.a1, self.a2, self.a3, self.a4
        if self.type == Limb.POLY:
            im -= ne.evaluate('ommu*(a1+ommu*(a2+ommu*(a3+ommu*a4)))')
        elif self.type == Limb.CLARET:
            im -= a1+a2+a3+a4
            msq = np.sqrt(mu)
            im += ne.evaluate('msq*(a1+msq*(a2+msq*(a3+msq*a4)))')
        return im

if __name__ == '__main__':

    import math as m
    import numpy as np
    import pylab as plt
    from trm import subs

    # Model: star 1 is on its own, stars 2 & 3 form
    # a tight binary in orbit with it. The line of
    # nodes of the star 1 / binary define the X axis
    # distance units here are solar radii

    s1 = 5.0 # surface brightness star 1
    r1 = 1.2 # radius star 1
    limb1 = Limb(Limb.POLY, 0.3)
    n1 = 100
    a1 = 104. # semi-major axis of its orbit around 1+2+3 CoM

    s2 = 1.0 # surface brightness star 2
    r2 = 0.30 # radius star 2
    limb2 = Limb(Limb.POLY, 0.3)
    n2 = 50
    a2 = 1.5 # semi-major axis of its orbit around 2+3 CoM

    s3 = 0.8 # surface brightness star 3
    r3 = 0.3 # radius star 3
    limb3 = Limb(Limb.POLY, 0.3)
    n3 = 50
    a3 = 2.0 # semi-major axis of its orbit around 2+3 CoM

    period1 = 210. # period of 1 + (2+3) orbit
    t1 = 0. # zero time of 1 + (2+3) orbit
    iangle1 = 89.8 # orbital inclination of 1 + (2+3) orbit

    a23 = 150. # semi-major axis of 2+3's orbit about 1+2+3
    iangle23 = 87. # inclination of 2+3 orbit
    Omega23 = 2.  # angle of line of nodes of 2+3 anti-c relative to 1
    period23 = 0.2585 # period of 2+3 orbit
    t23 = 0. # zero time of (2+3) orbit

    # convert angles to radians
    iangle1  = m.radians(iangle1)
    iangle23 = m.radians(iangle23)
    Omega23  = m.radians(Omega23)

    # axis vectors aligned with orbit 1 in terms of sky coords
    # x along line of nodes, z along orbital axis
    x1 = subs.Vec3(1,0,0)
    z1 = subs.Vec3(0,m.sin(iangle1),m.cos(iangle1))
    y1 = z1.cross(x1)

    x23 = subs.Vec3(m.cos(Omega23),m.sin(Omega23),0)
    z23 = subs.Vec3(-m.sin(iangle23)*m.sin(Omega23),
                    m.sin(iangle23)*m.cos(Omega23),m.cos(iangle23))
    y23 = z23.cross(x23)

    # array of times
    t = np.linspace(50,55,10000)

    # compute mean anomolies of each orbit
    # in this case assume circular so this is far enough
    phase1 = 2.*m.pi*(t-t1)/period1
    phase23 = 2.*m.pi*(t-t23)/period23

    cosp1 = np.cos(phase1)
    sinp1 = np.sin(phase1)
    cosp23 = np.cos(phase23)
    sinp23 = np.sin(phase23)

    f = []
    first = True
    for cp1, sp1, cp23, sp23 in zip(cosp1, sinp1, cosp23, sinp23):
        v1  = cp1*x1 + sp1*y1
        v2  = cp23*x23 + sp23*y23

        p1  =  a1*v1
        p23 = -a23*v1
        p2  = p23 + a2*v2
        p3  = p23 - a3*v2

        f1, f2, f3 = flux3(r1, r2, r3, limb1, limb2, limb3,
                           n1, n2, n3, p1, p2, p3)
        if first:
            total = s1*flux3.f1+s2*flux3.f2+s3*flux3.f3
            first = False

        f.append((s1*f1+s2*f2+s3*f3)/total)

    plt.plot(t,f)
    plt.show()

