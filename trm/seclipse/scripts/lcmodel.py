#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from trm import seclipse, orbits, cline

def lcmodel(args=None):
    """``lcmodel model dat (norm plot (pres (nbin)) | time1 time2 ntime texp ndiv)``

    Compute and plots the eclipse of limb-darkened spheres. It either
    does so given an already existing data file as a template or on a
    regularly-spaced set of times.

    Arguments::

      model : str
         the file with the model

      dat : str
         data file to read, 'none' if there are no data to be read.

      norm : bool [if data != 'none']
         True to normalise the data to give a minimum chi**2

      plot :  bool [if data != 'none']
         True to plot

      time1 : float [if data == 'none']
         First time to compute

      time2 : float [if data == 'none']
         Last time to compute

      ntime : int [if data == 'none']
         Number of time

      texp : float [if data == 'none']
         Exposure length

      ndiv : int [if data == 'none']
         Number of sub-division points per exposure
    """

    command, args = cline.script_args(args)

    # get the inputs
    with cline.Cline("SECLIPSE_ENV", ".seclipse", command, args) as cl:

        # register parameters
        cl.register('model', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('data', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('norm', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time1',cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time2', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('ntime', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('texp', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('ndiv', cline.Cline.LOCAL, cline.Cline.PROMPT)

        # get them
        mod = cl.get_value('model', 'light curve model', cline.Fname('lc', '.mod'))
        model = seclipse.model.Model(mod)

        dat  = cl.get_value(
            'data', "light curve data ['none' to sidestep]",
            cline.Fname('lc', '.dat'),
            ignore='none'
        )
        if dat is not None:
            # load data
            norm = cl.get_value('norm', 'normalise to minimize chi**2?', True)

        else:

            # create regularly spaced data
            time1 = inpt.get_value('time1', 'start time', 0.)
            time2 = inpt.get_value('time2', 'end time', 100.)
            ntime = inpt.get_value('ntime', 'number of times', 2)
            texp = inpt.get_value('texp', 'exposure time', 0.01, 0.)
            ndiv = inpt.get_value('ndiv', 'sub-division factor to smear exposures', 1, 1)

    # inputs obtained. do something

    if dat is not None:

        # Load data case
        ts,tes,fs,fes,ws,nds = seclipse.model.load_data(dat)
        fit = model.fit(ts,tes,nds)

        if norm:
            # normalise for minimum chi-squared
            sfac = seclipse.model.calc_sfac(fit, fs, fes, ws)
            fit *= sfac
            print('Scaled by',sfac)

            wgts = ws/fes**2
            chisq = (wgts*(fs-fit)**2).sum()
            print('Weighted chisq =',chisq)

        # rather specific time offset applied
        ts += 2454833 - 2400000 - 57240

        plt.plot(ts,fs,'.g')
        plt.plot(ts,fit,'r')
        plt.xlabel('Time [MJD]')
        plt.ylabel('Flux')
        plt.show()

    else:

        # no data case
        ts = np.linspace(time1, time2, ntime)
        tes = texp*np.ones_like(ts)
        nds = ndiv*np.ones_like(ts,dtype=np.int)
        t0 = time.time()
        fit = model.fit(ts,tes,nds)
        fit /= fit.max()

        # plot
        plt.plot(ts,fit,'b')
        plt.xlabel('Time [MJD - 58600]')
        plt.ylabel('Flux')
        plt.show()

