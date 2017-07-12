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
