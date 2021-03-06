#!/usr/bin/env python

# standard
import sys
import os
import math as m
import numpy as np
from multiprocessing import Pool
import importlib

# third party 
import emcee

# mine
from trm import cline, seclipse, pmcmc
from trm.cline import Cline

class Dmodel(seclipse.model.Model):
    """
    Class derived from seclipse.model.Model to define
    a model.
    """

    def __init__(
            self, arg,
            model_ok=None, model_adjust=None,
            model_prior=None, model_ranges=None
    ):

        super().__init__(arg)

        self.model_ok = model_ok
        self.model_adjust = model_adjust
        self.model_prior = model_prior

        if model_ranges is not None:
            assert len(model_ranges) >= len(self.vnames), \
                f'Too few parameter ranges specified ({len(model_ranges)})' + \
                f' cf number of variables ({len(self.vnames)})'

            # store lower/upper parameter limits
            self.plos = np.empty(len(self.vnames))
            self.phis = np.empty(len(self.vnames))
            for n, vname in enumerate(self.vnames):
                plo,phi = model_ranges[vname]
                self.plos[n] = plo
                self.phis[n] = phi
        else:
            self.plos = None

    def ok(self, verbose=False):
        """
        Checks are added to checks from Model
        """
        if super().ok():

            # first parameter range checks
            if self.plos is not None:
                for n, vname in enumerate(self.vnames):
                    pvalue = self[vname][0]
                    if pvalue < self.plos[n] or pvalue > self.phis[n]:
                        if verbose:
                            print(f'"{vname}" out of range: {self.plos[n]} to {self.phis[n]}')
                        return False

            # any additional checks
            if self.model_ok is not None and \
               not self.model_ok(self):
                if verbose:
                    print(f'model failed additional tests')
                return False
            return True
        else:
            if verbose:
                print('failed standard model check')
            return False

    def adjust(self):
        if super().adjust():
            if self.model_adjust is not None and \
               not self.model_adjust(self):
                return False
            else:
                return True
        else:
            return False

    def prior(self):
        if self.model_prior is not None:
            return super().prior() + self.model_prior(self)
        else:
            return super().prior()

__all__ = ['mcmc',]

class Logger:
    """
    Result logging class. Opens a file, writes some header and then
    provides a method to write data vectors to it. The basic data line
    is a walker plus a bunch of extras.
    """
    def __init__(self, lname, model, nwalker, nstore, dfile, soft, append):
        if append:
            self.fptr = open(lname,'a')
        else:
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

    """``mcmc model code data olog (nwalker sfac) nstore ntrial nthreads
    soft stretch log best''

    Carries out MCMC iterations of multi-sphere light curve model.

    This can (optionally) import a user-defined file which should
    include three methods, "model_ok", "model_adjust" and
    "model_prior" and a dictionary "model_ranges. "model_ok",
    "model_adjust" and "model_prior" each take an seclipse.model.Model
    as their only input. The first two both return True or False
    depending on whether they run OK. "model_ok" checks that the input
    model is valid; "model_adjust" alters fixed parameters according
    to the variable parameters, e.g. to implement Kepler's
    laws. "model_ranges" looks like {'r1' : (0.1, 0.5), 'r2' : (0.5,
    1.5)} i.e. it's a dictionary specifying upper and lower limits for
    each variable parameter for the model in question. "model_prior"
    returns a float to add to whatever prior is there by default to
    represent ln(prior) prob.

    Arguments::

       model : str
          if log is new, need to start with a model

       code : str
          file of code as explained above (model_ok etc). 'none'
          to ignore.

       data : str
          data file name

       olog : str
          old log file to initialise walkers. This will also define
          the number of walkers. "none" to ignore. This file will be
          appended to.

       nwalker : int [if olog == 'none']
          how many walkers per group. In this case, walkers are
          initialised by perturbing the starter model. It should
          be at least 2*(number variable parameters) to keep emcee
          happy.

       sfac : float [if olog == 'none']
          factor to scale perturbations by at start when initialising. If
          small, it avoids models wandering too far from where they can
          never be recovered.

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

       log : str if olog == 'none']
          log file to store results

       best : str
          file to save best model encountered to.

    """

    # First section is all about the input parameters, defining then
    # and getting their values.

    command, args = cline.script_args(args)

    with cline.Cline('SECLIPSE_ENV', '.seclipse', command, args) as cl:

        # register parameters
        cl.register('model', Cline.LOCAL, Cline.PROMPT)
        cl.register('code', Cline.LOCAL, Cline.PROMPT)
        cl.register('data', Cline.LOCAL, Cline.PROMPT)
        cl.register('olog', Cline.LOCAL, Cline.PROMPT)
        cl.register('nwalker', Cline.LOCAL, Cline.PROMPT)
        cl.register('sfac', Cline.LOCAL, Cline.PROMPT)
        cl.register('nstore', Cline.LOCAL, Cline.PROMPT)
        cl.register('ntrial', Cline.LOCAL, Cline.PROMPT)
        cl.register('nthreads', Cline.LOCAL, Cline.PROMPT)
        cl.register('soft', Cline.LOCAL, Cline.PROMPT)
        cl.register('stretch', Cline.LOCAL, Cline.PROMPT)
        cl.register('log', Cline.LOCAL, Cline.PROMPT)
        cl.register('best', Cline.LOCAL, Cline.PROMPT)

        # get them
        mod = cl.get_value(
            'model', 'light curve model', cline.Fname('lc', '.mod')
        )
        code = cl.get_value(
            'code',
            "python code to control model ['none' to ignore]",
            cline.Fname('code', '.py'), ignore='none'
        )[:-3]

        # Construct the model
        if code is None:
            # default
            model = Dmodel(mod)
        else:
            sys.path.append(".")
            code = importlib.import_module(code)
            model = Dmodel(
                mod, code.model_ok, code.model_adjust,
                code.model_prior, code.model_ranges
            )

        if not model.adjust() or not model.ok(True):
            print('Initial model fails parameter checks; please fix')
            exit(1)

        data = cl.get_value(
            'data', 'light curve data', cline.Fname('lc', '.dat')
        )
        t,te,f,fe,w,nd = seclipse.model.load_data(data)


        olog = cl.get_value(
            'olog',
            "old log file to initialise walkers ['none' to ignore]",
            cline.Fname('lc', '.log'), ignore='none'
        )
        if olog is None:
            nwalker = cl.get_value('nwalker', 'number of walkers', 100, 10)
            sfac = cl.get_value(
                'sfac', 'scaling factor for initialising walkers', 0.01, 1e-10
            )
        nstore = cl.get_value('nstore', 'how often to store results', 1, 1)
        ntrial = cl.get_value('ntrial', 'number of trials', 10000, 1)
        nthreads = cl.get_value('nthreads', 'number of threads', 1, 1)
        soft = cl.get_value(
            'soft', 'softening factor to scale chi**2 down', 1., 1.e-20
        )
        stretch = cl.get_value('stretch', 'stretch factor for emcee', 2.0, 1.1)

        if olog is None:
            log = cl.get_value(
                'log', 'MCMC log file',
                cline.Fname('lc', '.log', exist=False)
            )
            append = False
        else:
            log = olog
            append = True

        best = cl.get_value(
            'best', 'best light curve model',
            cline.Fname('save', '.mod', cline.Fname.NEW)
        )

    # OK, done with parameters.

    # Create ln(posterior) function object
    lnpost = Lnpost(model, t, te,f, fe, w, nd, soft)

    if olog is None:
        # Generate nwalker "walkers" to start emcee by randomly perturbing
        # around the starting model. We ensure the starting model is the
        # first one and that all models are initially viable, but give up
        # if we can't do so after trying 10x nwalker models.
        start, sigma = model.cvars()

        walkers = [start,]
        n = 1
        while len(walkers) < nwalker:
            p = np.random.normal(start,sfac*sigma)
            model.update(p)
            if model.adjust() and model.ok():
                walkers.append(p)

            n += 1
            if n > 10*nwalker:
                print(f'Tried {n} models but have only found {len(walkers)}')
                print('good ones. Giving up.')
                exit(1)
    else:
        # get walkers from old file
        ol = pmcmc.Log(olog)
        nwalker = ol.nwalker
        tab = ol.data[-nwalker:]
        cnames = tab.colnames[:-3]
        walkers = np.empty((nwalker,len(cnames)),dtype=np.float64)
        for n, cname in enumerate(cnames):
            walkers[:,n] = tab[cname].astype(np.float64)

    # Name & type the "blobs" (emcee terminology). Must match order
    # returned by Lnpost.__call__ (after the lnpost value)
    bdtype = [
        ('lnprior', float), ('chisq', float),
    ]
    lnpmax = -1.e30

    # Create log
    logger = Logger(log, model, nwalker, nstore, data, soft, append)

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
                
