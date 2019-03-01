#!/usr/bin/env python

import sys
import os
import math as m
import numpy as np
import emcee
from trm import subs
import trm.subs.input as inp
from trm import seclipse
from trm import mcmc

__all__ = ['mcmc',]

class Lnpost(object):
    """Function object that returns ln(post prob) for emcee given a vector of
    variable parameters"""

    def __init__(self, model, t, te, f, fe, w, nd, soft):
        """Parameters::

           model : (seclipse.model.Model)
              object representing a model

           t     : (array)
              times

           te    : (array)
              exposure times

           f     : (array)
              fluxes

           fe    : (array)
              errors on fluxes

           w     : (array)
              weights

           nd    : (array)
              sub-division factors

           soft  : (float)
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
        Computes weighted chi**2 corresponding to set of variable
        parameters specified in p which must as same order as
        returned by self.m.cvars
        """

        # update the model using the current parameters
        self.m.update(p)

        if self.m.adjust() and self.m.ok():
            prior = self.m.prior()
            if prior < 1.e20:

                # weighted chi**2
                chisq = self.m.chisq(
                    self.t,self.te,self.f,self.fe,self.w,self.nd,
                )

                lpost = (chisq/self.soft+prior)/2.
                return -lpost
            else:
                return -1.e30
        else:
            # fail parameter check, can return quickly with a value that will
            # never be selected
            return -1.e30

def mcmc(args=None):

    """``mcmc log prior (model nthreads nstore nwalker) data ntrial
    (sfac stretch) soft output''

    Carries out MCMC iterations of multi-star light curve model.

    By default uses no prior. To alter the prior write a file called
    "prior.py" in the directory where you run this. This should
    override the definition of the "prior" method of the
    seclipse.model.Model object used to define the triple / quadruple
    model using the derived class "Dmodel". e.g. The following code
    tacks on a constraint on the parameter 'a2' onto to whatever are
    already applied by the default "prior"::

    class Dmodel(Model):
        def prior(self):
            pri = super(Dmodel, self).prior()
            if self['a2'][0] > 3.:
                pri += ((self['a2'][0]-3.)/0.05)**2
            return pri

    This is imported if available.

    """


    # generate arguments
    if args is None:
        args = sys.argv.copy()

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('log', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('prior', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('nthreads', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('nstore', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('nwalker', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('data', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ntrial', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('sfac', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('stretch', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('soft', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('output', inp.Input.LOCAL, inp.Input.PROMPT)

    # get them
    log = inpt.get_value(
        'log', 'MCMC log file',
        subs.Fname('lc', '.log', exist=False)
    )

    # import prior
    prior = inpt.get_value(
        'prior', 'file containing definition of the prior',
        subs.Fname('prior', '.py', exist=False)
    )
    if os.path.exists(prior):
        __import__(prior)
    else:
        class Dmodel(seclipse.model.Model):
            pass

    if os.path.exists(log):
        # if a log file exists, read it in to find some of the parameters
        chain = mcmc.Chain(log)

        nwalker = chain.nwalker
        if len(chain) < nwalker:
            print('Too few walkers in',log,'; please delete and re-start.')
            exit(1)
        nstore = chain.nstore
        stretch = chain.stretch

        # Define initial model, fix order of variables
        model = Dmodel(chain.model)

        if not model.ok():
            print('Initial model fails parameter check in ok(); please fix')
            exit(1)

        walkers = []
        for i in range(nwalker):
            walkers.append(chain.vals[-1-i,:-3])
        append = True

    else:
        mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
        # Define initial model
        model = Dmodel(mod)
        if not model.ok():
            print('Initial model fails parameter check in ok(); please fix')
            exit(1)

        nstore = inpt.get_value('nstore', 'how often to store results', 1, 1)
        nwalker = inpt.get_value('nwalker', 'number of walkers', 100, 10)
        sfac = inpt.get_value('sfac', 'scaling factor to apply to initial scatter',
                              0.1, 1.e-4)
        stretch = inpt.get_value('stretch', 'stretch factor for emcee',
                                 2, 1.1)
        append = False

    # Load data
    dat = inpt.get_value('data', 'light curve data', subs.Fname('lc', '.dat'))
    t,te,f,fe,w,nd = seclipse.model.load_data(dat)

    # Remaining parameters
    nthreads = inpt.get_value('nthreads', 'number of threads', 1, 1)

    ntrial = inpt.get_value('ntrial', 'number of trials', 10000, 1)

    soft = inpt.get_value('soft', 'softening factor to scale chi**2 down',
                          1., 1.e-20)

    output = inpt.get_value('output', 'best light curve model',
                            subs.Fname('save', '.mod', subs.Fname.NEW))
    inpt.save()


    if not append:
        # In this case we need to generate nwalker models which we do by
        # randomly perturbing around starting model We ensure the starting
        # model is the first one and ensure that all models are initially
        # viable. 'sfac' controls the amount of perturbation.

        start, sigma = model.cvars()
        sigma *= sfac
        walkers = [start,]
        n = 1
        while n < nwalker:
            p = np.random.normal(start,sigma)
            model.update(p)
            if model.adjust() and model.ok():
                walkers.append(p)
                n += 1
                if n > 10*nwalker:
                    print('Tried > 10x nwalker models but have still not found')
                    print('nwalker =',nwalker,'good ones. Giving up. Sorry.')
                    exit(1)

    # Create Chisq function object
    lnpost = Lnpost(model, t, te,f, fe, w, nd, soft)

    # create sampler
    sampler = emcee.EnsembleSampler(
        nwalker, len(walkers[0]), lnpost, a=stretch, threads=nthreads
    )

    lnps = None
    rs = None
    lnpmax = -1.e30

    # create an MCMC compatible output file for results (or add to an old one,
    # assuming it is compatible)
    if append:
        oflag = 'a'
    else:
        oflag = 'w'

    with open(log,oflag) as fout:
        if not append:
            fout.write('## ' + ' '.join(model.vnames) + ' chisq lprior lpost\n')
            fout.write('#\n')
            fout.write('# Initial model:\n')
            fout.write('#\n')
            model.write(fout,'# ')
            fout.write('#\n')
            fout.write('# nstore  = ' + str(nstore) + '\n')
            fout.write('# method  = a\n')
            fout.write('# nwalker = ' + str(nwalker) + '\n')
            fout.write('# stretch = ' + str(stretch) + '\n')
            fout.write('# Minimum chi**2 = 1\n')
            fout.write('#\n')
            fout.write('# Jump start:\n')
            fout.write('#\n')
            fout.write('# RMS values:\n')
            fout.write('#\n')
            for vnam, sig in zip(model.vnames, sigma):
                fout.write('# ' + vnam + ' = ' + str(sig) + '\n')
            fout.write('#\n')
            fout.write('# Jump end:\n')
            fout.write('#\n')

        # let the trials begin
        for i in range(ntrial // nstore):

            walkers, lnps, rs = sampler.run_mcmc(walkers, nstore, rs, lnps)

            print('Group',i+1,'acceptance rate =',sampler.acceptance_fraction.mean())

            # check to see if we have improved the model
            if lnps.max() > lnpmax:
                lnpmax = lnps.max()
                best = walkers[np.argmax(lnps)]
                model.update(best)
                model.adjust()
                lnpri = model.prior()
                chisq = -soft*(2*lnpmax+lnpri)
                print('New best model. -2*ln(pprob),-2*ln(pri),chisq =',\
                          -2*lnpmax,lnpri,chisq)

                for name, value in zip(model.vnames, best):
                    print('{0:20s} = {1:s}'.format(name, str(value)))

                # write to output file
                with open(output,'w') as fsave:
                    fsave.write(
"""#
# File generated by mcmc.py
#
# Softening factor = {0:4.4g}
#
# chi**2 = {1:7.7g}, -2*ln(pri) = {2:7.7g}, -2*ln(post) = {3:7.7g}
#
""".format(soft,chisq,lnpri,2*lnpmax))
                    model.write(fsave)

            sampler.reset()

            # write out results
            for lnp, walker in zip(lnps, walkers):
                model.update(walker)
                lnpri = model.prior()
                chisq = -soft*(2*lnp+lnpri)
                strs  = ['{0:17.10e}'.format(item) for item in walker]
                fout.write(' '.join(strs) + ' {0:13.6e} {1:13.6e} {2:13.6e}\n'.format(chisq,lnpri,-lnp))
            fout.flush()

