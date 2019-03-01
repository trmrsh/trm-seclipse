#!/usr/bin/env python

"""
computes the eclipse of limb-darkened spheres, dumps the results to disk in
the same format as the standard input data files. It can either compute to
match an already existing data file or on a fine array of times.
"""

from __future__ import division

import sys
import math
import time
import numpy as np
import trm.subs.input as inp
from trm import seclipse, orbits, subs
from trm.subs import Vec3

if __name__ == '__main__':

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('output', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ldat',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('data',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('norm',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('time1', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('time2', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ntime', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('texp',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ndiv',  inp.Input.LOCAL, inp.Input.PROMPT)

    # get them
    mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
    model = seclipse.model.Model(mod)

    outfile = inpt.get_value('output', 'output synthetic data', subs.Fname('fake', '.dat', subs.Fname.NEW))

    ldat = inpt.get_value('ldat', 'load data file?', True)
    if ldat:
        dat  = inpt.get_value('data', 'light curve data', subs.Fname('lc', '.dat'))
        norm = inpt.get_value('norm', 'normalise to minimize chi**2?', True)
        inpt.save()

        # Load data
        ts,tes,fs,fes,ws,nds = seclipse.model.load_data(dat)

        fit = model.fit(ts,tes,nds)

        if norm:
            sfac = seclipse.model.calc_sfac(fit, fs, fes, ws)
            fit *= sfac
            print('Scaled by',sfac)

            wgts = ws/fes**2
            chisq = (wgts*(fs-fit)**2).sum()
            print('Weighted chisq =',chisq)

        header = """
This file was output by cmodel which calculated the data from the input
model file {:s} matching the data file {:s}

Columns are time, exposure time
(days), flux, error in flux, weighting factor
for chi**2, sub-division factor for exposure
smearing.
""".format(mod, data)

    else:
        time1 = inpt.get_value('time1', 'start time', 0.)
        time2 = inpt.get_value('time2', 'end time', 100.)
        ntime = inpt.get_value('ntime', 'number of times', 2)
        texp  = inpt.get_value('texp', 'exposure time', 0.01, 0.)
        ndiv  = inpt.get_value('ndiv', 'sub-division factor to smear exposures',
                               1, 1)
        inpt.save()

        ts  = np.linspace(time1, time2, ntime)
        tes = texp*np.ones_like(ts)
        ws = np.ones_like(ts)
        nds = ndiv*np.ones_like(ts,dtype=np.int)
        t0 = time.time()
        fit = model.fit(ts,tes,nds)
        fit /= fit.max()
        fes = 1.e-10*np.ones_like(ts)

        header = """
This file was output by cmodel which calculated the data from the input
model file {:s} using {:d} times spaced regularly from
{:f} to {:f}

Columns are time, exposure time
(days), flux, error in flux, weighting factor
for chi**2, sub-division factor for exposure
smearing.
""".format(mod, ntime, time1, time2)

    np.savetxt(outfile,np.column_stack([ts,tes,fit,fes,ws,nds]),
               '%.14e %.3e %.8e %.3e %.3f %d', header=header) 



