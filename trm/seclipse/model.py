from __future__ import division, print_function

"""
Sub-module to define particular stellar configurations (triples, quadruples)
for modelling light curves.
"""

import math
from collections import OrderedDict
import numpy as np
from trm import orbits, subs
from .core import Limb
from . import ring
import time

# first a few helper routines

def sol2au(length):
    """Converts from solar radii to AU"""
    return subs.RSUN*length/subs.AU

def load_data(dfile):
    """Loads a data file assumed to be in space-separated
    column form with columns of time, exposure time, flux,
    errors in flux, weights, sub-division factors (ints)

    Returns (ts,tes,fs,fes,ws,nds)
    """
    ts, tes, fs, fes, ws, nds = np.loadtxt(dfile, unpack=True)
    nds = nds.astype(np.int)
    print('Loaded',len(ts),'points from',dfile)
    return (ts,tes,fs,fes,ws,nds)

def write_data(fname, ts, tes, fs, fes, ws, nds, comment=''):
    """
    Writes out the data
    """

    header = """
This file was written by seclipse.model.write_data

Columns are time, exposure time (days), flux, error in flux, weighting factor
for chi**2, sub-division factor for exposure smearing.

""" + comment

    np.savetxt(fname, np.column_stack([ts,tes,fs,fes,ws,nds]),
               '%14.9f %9.3e %8.6f %8.6f %6.4f %2d', header=header)

def calc_sfac(fit, fs, fes, ws=None):
    """
    Computes optimum scaling factor given a fit = fit,
    to fluxes fs with errors fes, and weights ws.
    """
    if ws is None: ws = np.ones_like(fs)
    wgts = ws/fes**2
    return (wgts*fit*fs).sum()/(wgts*fit**2).sum()


class Model(dict):

    """Represents a stellar system, either a triple or quadruple star,
    modelled as a set of limb-darkened spheres in hierarchical Keplerian
    orbits.

    Two configurations are supported, representing a triple and a quadruple
    star. Which is used is determined using the parameter 'model' in the input
    data file used to define the model, which can either take the value
    'triple' or 'quad2'. Here are the definitions of the two models:


    1) Name = 'triple'

    Defines a light curve model for a triple star built up with a hierarchy of
    Kepler 2-body orbits (an approximation). Stars 1 & 2 form the first inner
    binary (1 / 2, binary 1), while stars (1+2) and star 2 forms the second
    outer binary ((1+2) / 3, binary 2).

    Thus there are two orbits to define as will be obvious below. Mostly this
    will be obvious through b1 = binary 1 etc, except for the semi-major axes
    where I think it is more memorable to try to use values that are
    associated with the stars where possible.

    Triple parameters::

      r1      : (float)
         radius of star 1 [solar]

      r2      : (float)
         radius of star 2 [solar]

      r3      : (float)
         radius of star 3 [solar]

      a1      : (float)
         semi-major axis of star 1 around binary 1's CoM [solar]

      a2      : (float)
         semi-major axis of star 2 around binary 1's CoM [solar]

      a3      : (float)
         semi-major axis of star 3 around system CoM [solar]

      ab      : (float)
         semi-major axis of binary 1 rel. to system CoM [solar]

      eb1     : (float)
         eccentricity of binary 1

      eb2     : (float)
         eccentricity of binary 2

      omegab1 : (float)
         argument of periapsis of binary 1 [degrees]

      omegab2 : (float)
         argument of periapsis of binary 2 [degrees]

      Pb1     : (float)
         period of binary 1 [days]

      Pb2     : (float)
         period of binary 2 [days]

      ib1     : (float)
         inclination of binary 1 [degrees]

      ib2     : (float)
         inclination of binary 2 [degrees]

      OMEGAb1 : (float)
         longitude of ascending node of binary 1 [degrees]

      OMEGAb2 : (float)
         longitude of ascending node of binary 2 [degrees]

      t0b1    : (float)
         zeropoint of binary 1 [days]

      t0b2    : (float)
         zeropoint of binary 2 [days]

      s1      : (float)
         surface brightness of star 1

      s2      : (float)
         surface brightness of star 2

      s3      : (float)
         surface brightness of star 3

      third   : (float)
         fractional "third" light, i.e. extra light from something other
         than stellar components. Range 0 to 1.

      limb1   : (float)
         linear limb darkening coeff of star 1

      limb2   : (float)
         linear limb darkening coeff of star 2

      limb3   : (float)
         linear limb darkening coeff of star 3

      n1      : (int)
         number of annuli over face of star 1

      n2      : (int)
         number of annuli over face of star 2

      n3      : (int)
         number of annuli over face of star 3

      ttype   : (int)
         type of t0. 1 = time of periastron. 2 = time of eclipse.
         The latter is more directly fixed by the data


    2) Name = 'quad2' representing a quadruple star.

    Defines a light curve model for a quadruple star built up with a hierarchy
    of Kepler 2-body orbits (an approximation). Stars 1 & 2 form the first,
    innermost, binary (1 / 2, binary 1), Stars (1+2) and star 4 forms the
    second, middle binary ((1+2) / 4, binary 2). Finally stars (1+2)+4 and
    star 3 forms the third outermost binary ((1+2)+4) / 3, binary 3). (The name
    'quad2' is used as there are two possible configurations of quadruples and
    'quad1' is reserved for the more obvious binary of two tight binaries.)

    Thus there are three orbits to define as will be obvious below. Mostly
    this will be obvious through b1 = binary 1 etc, except for the semi-major
    axes where I think it is more memorable to try to use values that are
    associated with the stars where possible.

    Quad2 parameters::

      r1      : (float)
         radius of star 1 [solar]

      r2      : (float)
         radius of star 2 [solar]

      r3      : (float)
         radius of star 3 [solar]

      r4      : (float)
         radius of star 4 [solar]

      a1      : (float)
         semi-major axis of star 1 around binary 1's CoM [solar]

      a2      : (float)
         semi-major axis of star 2 around binary 1's CoM [solar]

      a3      : (float)
         semi-major axis of star 3 around system CoM [solar]

      a4      : (float)
         semi-major axis of star 4 around binary 2's CoM [solar]

      ab1     : (float)
         semi-major axis of binary 1 rel. to binary 2's CoM [solar]

      ab2     : (float)
         semi-major axis of binary 2 rel. to system CoM [solar]

      eb1     : (float)
         eccentricity of binary 1

      eb2     : (float)
         eccentricity of binary 2

      eb3     : (float)
         eccentricity of binary 3

      omegab1 : (float)
         argument of periapsis of binary 1 [degrees]

      omegab2 : (float)
         argument of periapsis of binary 2 [degrees]

      omegab3 : (float)
         argument of periapsis of the ((1+2)+4) / 3 orbit [degrees]

      Pb1     : (float)
         period of binary 1 [days]

      Pb2     : (float)
         period of binary 2 [days]

      Pb3     : (float)
         period of binary 3 [days]

      ib1     : (float)
         inclination of binary 1 [degrees]

      ib2     : (float)
         inclination of binary 2 [degrees]

      ib3     : (float)
         inclination of binary 3 [degrees]

      OMEGAb1 : (float)
         longitude of ascending node of binary 1 [degrees]

      OMEGAb2 : (float)
         longitude of ascending node of binary 2 [degrees]

      OMEGAb3 : (float)
         longitude of ascending node of binary 3 [degrees]

      t0b1    : (float)
         zeropoint of binary 1 [days]

      t0b2    : (float)
         zeropoint of binary 2 [days]

      t0b3     : (float)
         zeropoint of binary 3 [days]

      s1      : (float)
         surface brightness of star 1

      s2      : (float)
         surface brightness of star 2

      s3      : (float)
         surface brightness of star 3

      s4      : (float)
         surface brightness of star 4

      third   : (float)
         fractional "third" light, i.e. extra light from something other
         than stellar components. Range 0 to 1.

      limb1   : (float)
         linear limb darkening coeff of star 1

      limb2   : (float)
         linear limb darkening coeff of star 2

      limb3   : (float)
         linear limb darkening coeff of star 3

      limb4   : (float)
         linear limb darkening coeff of star 4

      n1      : (int)
         number of annuli over face of star 1

      n2      : (int)
         number of annuli over face of star 2

      n3      : (int)
         number of annuli over face of star 3

      n4      : (int)
         number of annuli over face of star 4

      ttype   : (int)
         type of t0. 1 = time of periastron. 2 = time of eclipse.
         The latter is more directly fixed by the data

    """

    # dictionary of parameter name dictionaries keyed by the name of the model
    # 'triple', 'quad2'. For each parameter there is a 3-element tuple giving
    # the amount to vary it by when initialising a chain, along with upper and
    # lower bounds, the latter are used by the 'Anneal' method.

    PARAMS = {\
        'triple' :
            {'r1' : (0.02, 1.e-4, 20.),
             'r2' : (0.02, 1.e-4, 20.),
             'r3' : (0.02, 1.e-4, 20.),
             'a1' : (0.01, 0., 400.),
             'a2' : (0.01, 0., 400.),
             'a3' : (0.2,  0., 400.),
             'ab' : (0.2,  0., 400.),
             'eb1' : (0.01, 0., 0.999999),
             'eb2' : (0.01, 0., 0.999999),
             'omegab1' : (0.01, 0., 360.),
             'omegab2' : (0.01, 0., 360.),
             'Pb1' : (1.e-6, 0.2585, 0.2586),
             'Pb2' : (1.e-6, 0.3, 50.),
             'ib1' : (0.01, 85., 90.),
             'ib2' : (0.01, 85., 90.),
             'OMEGAb1' : (0.05, 170., 190.),
             'OMEGAb2' : (0.05, 170., 190.),
             't0b1' : (0.01, 0.,1000.),
             't0b2' : (0.01, 0.,1000.),
             's1' : (1., 0., 5000.),
             's2' : (1., 0., 5000.),
             's3' : (1., 0., 5000.),
             'third' : (0.05, 0., 1.),
             'limb1' : (0.01, 0., 1.),
             'limb2' : (0.01, 0., 1.),
             'limb3' : (0.01, 0., 1.),
             'n1' : (0, 2, 1000),
             'n2' : (0, 2, 1000),
             'n3' : (0, 2, 1000),
             'ttype' : (0, 1, 2),
             },

        'quad2' :
            {'r1' : (0.02, 1.e-4, 20.),
             'r2' : (0.02, 1.e-4, 20.),
             'r3' : (0.02, 1.e-4, 20.),
             'r4' : (0.02, 1.e-4, 20.),
             'a1' : (0.01, 0., 400.),
             'a2' : (0.01, 0., 400.),
             'a3' : (0.2,  0., 400.),
             'a4' : (0.2,  0., 400.),
             'ab1' : (0.2,  0., 400.),
             'ab2' : (0.2,  0., 400.),
             'eb1' : (0.01, 0., 0.999999),
             'eb2' : (0.01, 0., 0.999999),
             'eb3' : (0.01, 0., 0.999999),
             'omegab1' : (0.01, 0., 360.),
             'omegab2' : (0.01, 0., 360.),
             'omegab3' : (0.01, 0., 360.),
             'Pb1' : (1.e-6, 0.2585, 0.2586),
             'Pb2' : (1.e-6, 0.3, 50.),
             'Pb3' : (0.001, 200., 210.),
             'ib1' : (0.01, 85., 90.),
             'ib2' : (0.01, 85., 90.),
             'ib3' : (0.01, 75., 90.),
             'OMEGAb1' : (0.05, 170., 190.),
             'OMEGAb2' : (0.05, 170., 190.),
             'OMEGAb3' : (0.05, 170., 190.),
             't0b1' : (0.01, 0.,1000.),
             't0b2' : (0.01, 0.,1000.),
             't0b3' : (0.02, 0.,1000.),
             's1' : (1., 0., 5000.),
             's2' : (1., 0., 5000.),
             's3' : (1., 0., 5000.),
             's4' : (1., 0., 5000.),
             'third' : (0.05, 0., 1.),
             'limb1' : (0.01, 0., 1.),
             'limb2' : (0.01, 0., 1.),
             'limb3' : (0.01, 0., 1.),
             'limb4' : (0.01, 0., 1.),
             'n1' : (0, 2, 1000),
             'n2' : (0, 2, 1000),
             'n3' : (0, 2, 1000),
             'n4' : (0, 2, 1000),
             'ttype' : (0, 1, 2),
             }
        }

    def __init__(self, arg):
        """Given a file with lines like:

        model = triple # model type
        r1 = 2.5 v  # variable parameter
        r2 = 1.0 f  # fixed parameter
        n1 = 12 # integer parameter

        (order immaterial) or an equivalent OrderedDict: {'model' : 'triple',
        ...} this defines the model parameters. Each of these will be loaded
        into a dictionary with two-element list values of the form [param,
        variable] where param is the parameter value and variable = True if
        variable.  The parameters are checked against a list called
        Model.PARAMS so print these out to see them all. The model type
        ('triple' or 'quad2') is stored separately as an attribute
        'model'. Other attributes are 'pnames' to store all parameter names
        and 'vnames' to store all variable parameter names.

        See the main help on Model objects for the full list of parameters.
        """

        self.pnames = []
        self.vnames = []
        if isinstance(arg, str):
            # Read in model from a file
            with open(arg) as fin:
                for line in fin:
                    if not line.startswith('#') and not line.isspace() and \
                            line != '' and line.find('=') > -1:
                        equals = line.find('=')
                        name = line[:equals].strip()

                        rest = line[equals+1:]
                        hash = line.find('#')
                        elems = rest[:hash].split() \
                            if hash > -1 else rest.split()

                        if name == 'model':
                            # Trap the model name
                            if elems[0] not in Model.PARAMS:
                                raise Exception(
                                    'Model type not recognised in line: {:s}'.format(line)
                                    )
                            self.model = elems[0]

                        elif len(elems) == 2:
                            if elems[1] == 'f':
                                # fixed floating point parameter
                                self[name] = [float(elems[0]),False]
                                self.pnames.append(name)
                            elif elems[1] == 'v':
                                # fixed floating point parameter
                                self[name] = [float(elems[0]),True]
                                self.vnames.append(name)
                                self.pnames.append(name)
                            else:
                                raise Exception(
                                    'Must specify "f"=fixed or "v"=variable after all float parameters, line: {:s}'.format(line)
                                    )
                        elif len(elems) == 1:
                            self[name] = [int(elems[0]),False]
                            self.pnames.append(name)
                        else:
                            raise Exception(
                                'Could not interpret line: {:s}'.format(line)
                                )

        elif isinstance(arg, OrderedDict):
            # Read in model from a dictionary (as may be created from an mcmc log file)
            for name, value in arg.items():
                elems = value.split()

                if name == 'model':
                    # Trap the model name
                    if elems[0] not in Model.PARAMS:
                        raise Exception(
                            'Model type not recognised in line: {:s}'.format(elems[0])
                            )
                    self.model = elems[0]

                elif len(elems) == 2:
                    if elems[1] == 'f':
                        # fixed floating point parameter
                        self[name] = [float(elems[0]),False]
                        self.pnames.append(name)
                    elif elems[1] == 'v':
                        # fixed floating point parameter
                        self[name] = [float(elems[0]),True]
                        self.vnames.append(name)
                        self.pnames.append(name)
                    else:
                        raise Exception(
                            'Must specify "f"=fixed or "v"=variable after all float parameters, line: {:s}'.format(line)
                                    )
                elif len(elems) == 1:
                    self[name] = [int(elems[0]),False]
                    self.pnames.append(name)
                else:
                    raise Exception(
                        'Could not interpret name / value: {:s} / {:s}'.format(name,value)
                        )

        else:
            raise Exception(
                'Argument was not a string [filename] or a dictionary'
                )

        # backwards compatibility
        if 'third' not in self.pnames:
            self['third'] = [0.,False]
            self.pnames.append('third')

        # check that all the expected parameters are defined
        for pname in Model.PARAMS[self.model]:
            if pname not in self.pnames:
                raise Exception(
                    'Model = {:s} parameter = {:s} is undefined'.format(self.model, pname)
                    )

        # check that no unexpected parameters are defined
        for pname in self.pnames:
            if pname not in Model.PARAMS[self.model]:
                raise Exception(
                    'Parameter = {:s} not recognised for model = {:s}'.format(pname, self.model)
                    )

    def paths(self, times):
        """
        Returns either 3x or 4x 3-element tuples containing x,y,z arrays
        representing the positions of each star corresponding to the input
        array of times.
        """

        # Unit conversions:
        #
        # angles: degrees to radians,
        # lengths: solar to AU

        if self.model == 'triple':
            omegab1 = math.radians(self['omegab1'][0])
            omegab2 = math.radians(self['omegab2'][0])
            OMEGAb1 = math.radians(self['OMEGAb1'][0])
            OMEGAb2 = math.radians(self['OMEGAb2'][0])
            ib1     = math.radians(self['ib1'][0])
            ib2     = math.radians(self['ib2'][0])
            a1      = sol2au(self['a1'][0])
            a2      = sol2au(self['a2'][0])
            a3      = sol2au(self['a3'][0])
            ab      = sol2au(self['ab'][0])

            return orbits.model.triplePos(
                times, a1, a2, a3, ab, self['eb1'][0],
                self['eb2'][0], omegab1, omegab2,
                self['Pb1'][0], self['Pb2'][0],
                ib1, ib2, OMEGAb1, OMEGAb2,
                self['t0b1'][0], self['t0b2'][0], self['ttype'][0],
                )

        elif self.model == 'quad2':
            omegab1 = math.radians(self['omegab1'][0])
            omegab2 = math.radians(self['omegab2'][0])
            omegab3 = math.radians(self['omegab3'][0])
            OMEGAb1 = math.radians(self['OMEGAb1'][0])
            OMEGAb2 = math.radians(self['OMEGAb2'][0])
            OMEGAb3 = math.radians(self['OMEGAb3'][0])
            ib1     = math.radians(self['ib1'][0])
            ib2     = math.radians(self['ib2'][0])
            ib3     = math.radians(self['ib3'][0])
            a1      = sol2au(self['a1'][0])
            a2      = sol2au(self['a2'][0])
            a3      = sol2au(self['a3'][0])
            a4      = sol2au(self['a4'][0])
            ab1     = sol2au(self['ab1'][0])
            ab2     = sol2au(self['ab2'][0])

            return orbits.model.quad1Pos(
                times, a1, a2, a3, a4, ab1, ab2, self['eb1'][0],
                self['eb2'][0], self['eb3'][0], omegab1, omegab2,
                omegab3, self['Pb1'][0], self['Pb2'][0], self['Pb3'][0],
                ib1, ib2, ib3, OMEGAb1, OMEGAb2, OMEGAb3, self['t0b1'][0],
                self['t0b2'][0], self['t0b3'][0], self['ttype'][0],
                )

        else:
            raise Exception(
                'Model = {:s} not recognised'.format(self.model)
                )

    def prior(self):
        """
        Returns -2*ln(prior) [suitable for adding to chisq]  based
        upon timing model. This is over-ridden in derived class used
        to re-define the prior through an input file.
        """
        return 0.

    def chisq(self, ts, tes, fs, fes, ws, nds):
        """
        Computes chi**2 of model given times, exposures etc. This
        also sets the maximum value of the fit (attribute 'max').
        It autoscales to give the minimum chi**2.

        Arguments::

          ts : array
             times

          tes : array
             exposure times

          fs : array
             fluxes

          fes : array
             error in fluxes

          ws : array
             weights

          nds : array
             integer sub-division factors

          ncpu : int
             number of CPUs to use
        """

        # autoscale to minimise chi**2
        fit  = self.fit(ts,tes,nds)
        wgt  = ws/fes**2
        sfac = (wgt*fit*fs).sum()/(wgt*fit**2).sum()
        chisq = (wgt*(fs-sfac*fit)**2).sum()
        self.max = fit.max()
        return chisq

    def fit(self, ts, tes, nds):
        """
        Computes light curves corresponding to input times.

        Arguments::

          ts   : (array)
             mid-times [days]

          tes  : (array)
             exposure times [days]

          nds  : (array)
             integer sub-division factors to smear exposures.
        """

        third = self['third'][0]

        if self.model == 'triple':

            limb1 = Limb(Limb.POLY, self['limb1'][0])
            limb2 = Limb(Limb.POLY, self['limb2'][0])
            limb3 = Limb(Limb.POLY, self['limb3'][0])

            s1 = self['s1'][0]*(subs.RSUN/subs.AU)**2
            s2 = self['s2'][0]*(subs.RSUN/subs.AU)**2
            s3 = self['s3'][0]*(subs.RSUN/subs.AU)**2

            r1 = sol2au(self['r1'][0])
            r2 = sol2au(self['r2'][0])
            r3 = sol2au(self['r3'][0])

            # Calculate arrays of annuli (radius and flux contribution) for
            # each sphere as an input to seclipse.rings.flux4
            rings1, fluxes1, tflux1 = ring.rfinit(r1, limb1, self['n1'][0])
            rings2, fluxes2, tflux2 = ring.rfinit(r2, limb2, self['n2'][0])
            rings3, fluxes3, tflux3 = ring.rfinit(r3, limb3, self['n3'][0])

            # Build array-like arguments for flux3, one element per sphere
            r = (r1,r2,r3)
            rings = (rings1,rings2,rings3)
            fluxes = (fluxes1,fluxes2,fluxes3)
            tflux = (tflux1,tflux2,tflux3)

            # Calculate positions of stellar CoMs at 'expanded' times
            # to allow for exposure smearing
            tnew = ring.expand(ts,tes,nds)
            p1s, p2s, p3s = self.paths(tnew)

            # Calculate light curve
            lnew = ring.lc3(r, rings, fluxes, tflux, s1, s2, s3, p1s, p2s, p3s)
            lc = ring.compress(lnew, nds)

            # add "third" light
            total = s1*tflux1+s2*tflux2+s3*tflux3
            lc = third*total + (1-third)*lc

        elif self.model == 'quad2':
            limb1 = Limb(Limb.POLY, self['limb1'][0])
            limb2 = Limb(Limb.POLY, self['limb2'][0])
            limb3 = Limb(Limb.POLY, self['limb3'][0])
            limb4 = Limb(Limb.POLY, self['limb4'][0])

            s1 = self['s1'][0]*(subs.RSUN/subs.AU)**2
            s2 = self['s2'][0]*(subs.RSUN/subs.AU)**2
            s3 = self['s3'][0]*(subs.RSUN/subs.AU)**2
            s4 = self['s4'][0]*(subs.RSUN/subs.AU)**2

            r1 = sol2au(self['r1'][0])
            r2 = sol2au(self['r2'][0])
            r3 = sol2au(self['r3'][0])
            r4 = sol2au(self['r4'][0])

            # Calculate arrays of annuli (radius and flux contqribution) for
            # each sphere as an input to seclipse.ring.flux4
            rings1, fluxes1, tflux1 = ring.rfinit(r1, limb1, self['n1'][0])
            rings2, fluxes2, tflux2 = ring.rfinit(r2, limb2, self['n2'][0])
            rings3, fluxes3, tflux3 = ring.rfinit(r3, limb3, self['n3'][0])
            rings4, fluxes4, tflux4 = ring.rfinit(r4, limb4, self['n4'][0])

            # Build array-like arguments for flux4, one element per sphere
            r = (r1,r2,r3,r4)
            rings = (rings1,rings2,rings3,rings4)
            fluxes = (fluxes1,fluxes2,fluxes3,fluxes4)
            tflux = (tflux1,tflux2,tflux3,tflux4)

            # Calculate positions of stellar CoMs
            tnew = ring.expand(ts,tes,nds)
            p1s, p2s, p3s, p4s = self.paths(tnew)

            # Calculate light curve
            lnew = ring.lc4(r, rings, fluxes, tflux, s1, s2, s3, s4, p1s, p2s, p3s, p4s)
            lc = ring.compress(lnew, nds)

            # add "third" light
            total = s1*tflux1+s2*tflux2+s3*tflux3+s4*tflux4
            lc = third*total + (1-third)*lc

        else:
            raise Exception(
                'Model = {:s} not recognised'.format(self.model)
                )

        return lc

    def cvars(self):
        """
        Returns arrays of values and typical spreads of the variable
        parameters in the order defined by the vnames attribute
        """
        vals   = []
        sigmas = []
        for name in self.vnames:
            vals.append(self[name][0])
            sigmas.append(Model.PARAMS[self.model][name][0])
        return (np.array(vals), np.array(sigmas))

    def update(self,p):
        """
        Updates variable parameters of a model given a vector or values.
        It is assumed that the values in p match the order of the vnames
        attribute. Woe betide you if this is not the case!
        """

        for n, name in enumerate(self.vnames):
            self[name][0] = p[n]

    def ok(self):
        """
        Carries out crude checks of parameter values. Comes back with False if
        there is a problem. These checks are not exhaustive and don't address
        dynamical issues.
        """

        if self.model == 'triple':
            flag = \
                self['r1'][0] > 0 and self['r2'][0] > 0 and \
                self['r3'][0] > 0 and \
                self['a1'][0] > 0 and self['a2'][0] > 0 and \
                self['a3'][0] > 0 and self['ab'][0] > 0 and \
                self['r1'][0] + self['r2'][0] < (self['a1'][0] + self['a2'][0])*(1-self['eb1'][0]) and \
                self['eb1'][0] >= 0. and self['eb1'][0] < 1. and \
                self['eb2'][0] >= 0. and self['eb2'][0] < 1. and \
                self['Pb1'][0] > 0. and self['Pb2'][0] > 0. and \
                self['s1'][0] >= 0. and self['s2'][0] >= 0. and \
                self['s3'][0] >= 0. and \
                self['third'][0] >= 0. and self['third'][0] <= 1. and \
                self['limb1'][0] >= 0. and self['limb1'][0] <= 1. and \
                self['limb2'][0] >= 0. and self['limb2'][0] <= 1. and \
                self['limb3'][0] >= 0. and self['limb3'][0] <= 1.

        elif self.model == 'quad2':
            flag = \
                self['r1'][0] > 0 and self['r2'][0] > 0 and \
                self['r3'][0] > 0 and self['r4'][0] > 0 and \
                self['a1'][0] > 0 and self['a2'][0] > 0 and \
                self['a3'][0] > 0 and self['a4'][0] > 0 and \
                self['ab1'][0] > 0 and self['ab2'][0] > 0 and \
                self['r1'][0] + self['r2'][0] < (self['a1'][0] + self['a2'][0])*(1-self['eb1'][0]) and \
                max(self['r1'][0] + self['a1'][0]*(1+self['eb1'][0]), self['r2'][0] + self['a2'][0]*(1+self['eb1'][0])) + \
                self['r4'][0] < (self['ab1'][0] + self['a4'][0])*(1-self['eb2'][0]) and \
                self['r3'][0] + self['r4'][0] < (self['ab2'][0] + self['a4'][0])*(1-self['eb3'][0]) and \
                self['eb1'][0] >= 0. and self['eb1'][0] < 1. and \
                self['eb2'][0] >= 0. and self['eb2'][0] < 1. and \
                self['eb3'][0] >= 0. and self['eb3'][0] < 1. and \
                self['Pb1'][0] > 0. and self['Pb2'][0] > 0. and self['Pb3'][0] > 0. and \
                self['s1'][0] >= 0. and self['s2'][0] >= 0. and \
                self['s3'][0] >= 0. and self['s4'][0] >= 0. and \
                self['third'][0] >= 0. and self['third'][0] <= 1. and \
                self['limb1'][0] >= 0. and self['limb1'][0] <= 1. and \
                self['limb2'][0] >= 0. and self['limb2'][0] <= 1. and \
                self['limb3'][0] >= 0. and self['limb3'][0] <= 1. and \
                self['limb4'][0] >= 0. and self['limb4'][0] <= 1.

        else:
            raise Exception(
                'Model = {:s} not recognised'.format(self.model)
                )
        return flag


    def write(self,fobj,prefix=''):
        """
        Writes  model parameters to a file object fobj, with an
        optional prefix
        """
        fobj.write(prefix + 'model = ' + self.model + '\n')
        for name in self.pnames:
            v = self[name]
            if isinstance(v[0],int):
                fobj.write(prefix + name + ' = ' + str(v[0]) + '\n')
            else:
                if v[1]:
                    fobj.write(prefix + name + ' = ' + str(v[0]) + ' v\n')
                else:
                    fobj.write(prefix + name + ' = ' + str(v[0]) + ' f\n')
