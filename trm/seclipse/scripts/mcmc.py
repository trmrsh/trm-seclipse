#!/usr/bin/env python

# standard
import sys
import os
import math as m
import numpy as np
from multiprocessing import Pool

# third party 
import emcee

# mine
from trm import cline, seclipse
from trm.cline import Cline

try:
    # Import Dmodel defined in Prior.py in directory
    # we are working from.
    sys.path.insert(0,'.')
    from Prior import Dmodel
    default = False
except:
    class Dmodel(seclipse.model.Model):
        pass
    default = True

__all__ = ['mcmc',]

class Logger:
    """
    Result logging class. Opens a file, writes some header and then
    provides a method to write data vectors to it. The basic data line
    is a walker plus a bunch of extras.
    """
    def __init__(self, lname, model, nwalker, nstore, dfile, soft):
        self.fptr = open(lname,'w')
        self.fptr.write(f'# nwalker = {nwalker}\n')
        self.fptr.write(f'# nstore = {nstore}\n')
        self.fptr.write(f'# dfile = {dfile}\n')
        self.fptr.write(f'# soft = {soft}\n')
        self.fptr.write(f'#\n')
        model.write(self.fptr,'# ')
        self.fptr.write("# Column names:\n")

        # column names
        cnames = model.vnames + ['lnpost','lnprior','chisq']
        self.fptr.write(f"{' '.join(cnames)}\n")

    def add_line(self, values, fmt='.10e'):
        for val in values:
            self.fptr.write(f'{val:{fmt}} ')
        self.fptr.write('\n')
        self.fptr.flush()

class Lnpost(object):
    """Function object that returns ln(post prob) for emcee given a vector of
    variable parameters"""

    def __init__(self, model, t, te, f, fe, w, nd, soft):
        """Parameters::

           model : seclipse.model.Model
              object representing a model

           t : array
              times

           te : array
              exposure times

           f : array
              fluxes

           fe : array
              errors on fluxes

           w : array
              weights

           nd : array
              sub-division factors

           soft : float
              softening factor used to downweight chi**2, equivalent to
              scaling up errors by sqrt(soft).

        """
        self.m = model
        self.t = t
        self.te = te
        self.f = f
        self.fe = fe
        self.w = w
        self.nd = nd
        self.soft = soft

    def __call__(self, p):
        """
        Returns ln(posterior), ln(prior), chisq
        """

        # update the model using the current parameters
        self.m.update(p)

        if self.m.adjust() and self.m.ok():
            lnprior = self.m.prior()

            # weighted chi**2
            chisq = self.m.chisq(
                self.t,self.te,self.f,self.fe,self.w,self.nd,
            )
            lnpost = lnprior - chisq/self.soft/2.
            return lnpost, lnprior, chisq
        else:
            return -np.inf, -np.inf, np.inf

def mcmc(args=None):

    """``mcmc model data nwalker nstore ntrial nthreads soft stretch log best''

    Carries out MCMC iterations of multi-sphere light curve model.

    This will import a file called "Prior.py" if it exists in the
    directory within which this is run. This should override the
    definition of the "prior" method of the seclipse.model.Model
    object used to define the triple / quadruple model using the
    derived class "Dmodel". e.g. the following code tacks on a
    constraint on the parameter 'a2' onto to whatever are already
    applied by the default "prior"::

    class Dmodel(Model):
        def prior(self):
            pri = super(Dmodel, self).prior()
            if self['a2'][0] > 3.:
                pri += ((self['a2'][0]-3.)/0.05)**2
            return pri

    This is imported if available.

    Arguments::

       model : str
          if log is new, need to start with a model

       data : str
          data file name

       nwalker : int
          how many walkers per group

       nstore : int
          how often to store results

       ntrial : int
          total number of groups (only 1 in nstore of will be stored)

       nthreads : int
          number of threads to run in parallel

       soft : float
          softening factor to divide into chi**2

       stretch : float
          emcee stretch factor

       dlnpmax : float
          maximum drop in ln(post) to accept when setting up walkers
          with respect to initial model. This is to reduce chances of
          totally wrong models which then never get removed. Don't make
          it too small or you won't get a starter set.

       log : str
          log file to store results (can be old)

       best : str
          file to save best model encountered to.
    """

    # First section is all about the input parameters, defining then
    # and getting their values.

    command, args = cline.script_args(args)

    with cline.Cline('SECLIPSE_ENV', '.seclipse', command, args) as cl:

        # register parameters
        cl.register('model', Cline.LOCAL, Cline.PROMPT)
        cl.register('data', Cline.LOCAL, Cline.PROMPT)
        cl.register('nwalker', Cline.LOCAL, Cline.PROMPT)
        cl.register('nstore', Cline.LOCAL, Cline.PROMPT)
        cl.register('ntrial', Cline.LOCAL, Cline.PROMPT)
        cl.register('nthreads', Cline.LOCAL, Cline.PROMPT)
        cl.register('soft', Cline.LOCAL, Cline.PROMPT)
        cl.register('stretch', Cline.LOCAL, Cline.PROMPT)
        cl.register('dlnpmax', Cline.LOCAL, Cline.PROMPT)
        cl.register('log', Cline.LOCAL, Cline.PROMPT)
        cl.register('best', Cline.LOCAL, Cline.PROMPT)

        # get them
        mod = cl.get_value(
            'model', 'light curve model', cline.Fname('lc', '.mod')
        )
        model = Dmodel(mod)
        if not model.ok():
            print('Initial model fails parameter check in ok(); please fix')
            exit(1)

        data = cl.get_value(
            'data', 'light curve data', cline.Fname('lc', '.dat')
        )
        t,te,f,fe,w,nd = seclipse.model.load_data(data)

        nwalker = cl.get_value('nwalker', 'number of walkers', 100, 10)
        nstore = cl.get_value('nstore', 'how often to store results', 1, 1)
        ntrial = cl.get_value('ntrial', 'number of trials', 10000, 1)
        nthreads = cl.get_value('nthreads', 'number of threads', 1, 1)
        soft = cl.get_value(
            'soft', 'softening factor to scale chi**2 down', 1., 1.e-20
        )
        stretch = cl.get_value('stretch', 'stretch factor for emcee', 2.0, 1.1)
        dlnpmax = cl.get_value(
            'dlnpmax', 'maximum difference in ln(post)', 1000., 10.
        )

        log = cl.get_value(
            'log', 'MCMC log file',
            cline.Fname('lc', '.log', exist=False)
        )

        best = cl.get_value(
            'best', 'best light curve model',
            cline.Fname('save', '.mod', cline.Fname.NEW)
        )

    # OK, done with parameters.

    if default:
        print('Using default prior')
    else:
        print('Using prior defined in Prior.py')

    # Create ln(posterior) function object
    lnpost = Lnpost(model, t, te,f, fe, w, nd, soft)

    # Generate nwalker "walkers" to start emcee by randomly perturbing
    # around the starting model. We ensure the starting model is the
    # first one and that all models are initially viable, but give up
    # if we can't do so after trying 10x nwalker models.

    start, sigma = model.cvars()
    lpstart,_d,_d = lnpost(start)

    walkers = [start,]
    n = 1
    while n < nwalker:
        p = np.random.normal(start,sigma)
        model.update(p)
        if model.adjust() and model.ok():
            lp,_d,_d= lnpost(p)
            if lpstart - lp < dlnpmax:
                walkers.append(p)
                n += 1
                if n > 10*nwalker:
                    print('Tried > 10x nwalker models but have still not found')
                    print('nwalker =',nwalker,'good ones. Giving up. Sorry.')
                    exit(1)

    # Name & type the "blobs" (emcee terminology). Must match order
    # returned by Lnpost.__call__ (after the lnpost value)
    bdtype = [
        ('lnprior', float), ('chisq', float),
    ]
    lnpmax = -1.e30

    # Create log
    logger = Logger(log, model, nwalker, nstore, data, soft)

    with Pool(nthreads) as pool:

        # create the sampler
        sampler = emcee.EnsembleSampler(
            nwalker, len(walkers[0]), lnpost, pool,
            emcee.moves.StretchMove(stretch),
            blobs_dtype=bdtype
        )

        nstep = 0
        for sample in sampler.sample(walkers, iterations=ntrial//nstore, thin_by=nstore):
            nstep += 1
            print(f'Group {nstep}, acceptance rate = {sampler.acceptance_fraction.mean()}')

            # extract all the values we want.
            lnps = sampler.get_log_prob()[-1]
            blobs = sampler.get_blobs()[-1]
            results = sampler.get_chain()[-1]

            # check to see if we have improved the model
            if lnps.max() > lnpmax:
                imax = lnps.argmax()
                lnpmax = lnps.max()

                # Crow about it
                print(f"New best model has ln(post) = {lnpmax}, chisq = {blobs['chisq'][imax]}")

                # Save the wonderful model to disk
                bres = results[imax]
                model.update(bres)
                model.adjust()

                # write to output file
                with open(best,'w') as fsave:
                    fsave.write(f"""#
# File generated by mcmc.py
#
# Softening factor = {soft}
#
# ln(post)  = {lnpmax}
# ln(prior) = {blobs['lnprior'][imax]}
# chi**2    = {blobs['chisq'][imax]}
#
""")
                    model.write(fsave)

            sys.stdout.flush()

            # Write the current walkers to a file, one line per walker
            for walker, lnp, lnprior, chisq in \
                zip(results, lnps, blobs["lnprior"], blobs["chisq"]):
                logger.add_line(np.concatenate((walker, [lnp, lnprior, chisq])))
                
