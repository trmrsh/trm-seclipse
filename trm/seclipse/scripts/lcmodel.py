#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from trm import seclipse, orbits, cline

def lcmodel(args=None):
    """``lcmodel model dat (norm plot (pres (nbin)) | time1 time2 ntime texp ndiv)
    (ppath) [save (reject dout)]``

    Compute and plots the eclipse of limb-darkened spheres. It either
    does so given an already existing data file as a template or on a
    regularly-spaced set of times. In the latter case it will also
    plot a representation of the paths of the spheres.

    Arguments::

      model : str
         the file with the model

      dat : str
         data file to read, 'none' if there are no data to be read.

      norm : bool (if data != 'none')
         True to normalise the data to give a minimum chi**2

      plot :  bool (if data != 'none')
         True to plot

    """

    command, args = cline.script_args(args)

    # get the inputs
    with cline.Cline("SECLIPSE_ENV", ".seclipse", command, args) as cl:

        # register parameters
        cl.register('model', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('data', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('norm', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time1',cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('plot', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('pres', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('nbin', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time2', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('ntime', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('texp', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('ndiv', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('ppath', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('sdata', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('reject',cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('dout', cline.Cline.LOCAL, cline.Cline.PROMPT)

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
            plot  = cl.get_value('plot', 'plot? (else just report chi**2)', True)
            if plot:
                pres = cl.get_value('pres', 'plot residuals?', True)
                if pres:
                    nbin = cl.get_value(
                        'nbin', 'number of bins for residuals (0 to ignore)', 1000, 0
                    )
                ppath = cl.get_value('ppath', 'plot paths?', True)
            else:
                ppath = False

            sdata = cl.get_value('sdata', 'save data with rejection?', True)
            if sdata:
                reject = cl.get_value('reject', 'rejection threshold', 5., 0.)
                dout = cl.get_value(
                    'dout', 'name of output file', cline.Fname('lc', '.dat', cline.Fname.NEW)
                )

        else:

            # create regularly spaced data
            time1 = inpt.get_value('time1', 'start time', 0.)
            time2 = inpt.get_value('time2', 'end time', 100.)
            ntime = inpt.get_value('ntime', 'number of times', 2)
            texp = inpt.get_value('texp', 'exposure time', 0.01, 0.)
            ndiv = inpt.get_value('ndiv', 'sub-division factor to smear exposures', 1, 1)
            smod = inpt.get_value('smod', 'save model?', True)
            if smod:
                dout = inpt.get_value(
                    'dout', 'name of output file',
                    subs.Fname('lc', '.dat', subs.Fname.NEW)
                )
            plot  = True
            ppath = cl.get_value('ppath', 'plot paths?', True)

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

        if sdata:
            scale = np.sqrt((((fs-fit)/fes)**2).sum()/len(fs))
            rej = np.abs((fs-fit)/fes) > scale*reject
            header = """
This file was output by lcmodel after rejecting bad data

Columns are time (BJD-2454833), exposure time (days), flux, error in flux,
weighting factor for chi**2, sub-division factor for exposure smearing.

            """
            np.savetxt(
                dout,
                np.column_stack(
                    [ts[~rej], tes[~rej], fs[~rej],
                     fes[~rej], ws[~rej], nds[~rej]]
                ),
                '%14.9f %9.3e %8.6f %8.6f %4.2f %2d',
                header=header
            )

        if plot:
            ts += 2454833 - 2400000 - 57240

            plt.plot(ts,fs,'.g')
            plt.plot(ts,fit,'r')
            if pres:
                if nbin:
                    tbins = np.linspace(ts.min(),ts.max(),nbin)
                    tbin,d = np.histogram(ts,tbins,weights=ts)
                    fbin,d = np.histogram(ts,tbins,weights=fs-fit)
                    nbin,d = np.histogram(ts,tbins)
                    ok = nbin > 0
                    tbin = tbin[ok]/nbin[ok]
                    fbin = fbin[ok]/nbin[ok]
                    plt.plot(tbin,fbin+0.92,'.b')
                else:
                    plt.plot(ts,fs-fit+0.88,'.b')

                if sdata:
                    plt.plot(ts[rej],fs[rej]-fit[rej]+0.88,'or')

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

    if plot:
        #            ts += 2454833 - 2400000 - 58600
        #            ts += 2454833 - 2400000
        plt.plot(ts,fit,'b')
        plt.xlabel('Time [MJD - 58600]')
        plt.ylabel('Flux')
        plt.show()

        if sdata:
            header = """
This is a model output

Columns are time (BJD-2454833), exposure time (days), flux

"""
            np.savetxt(
                dout,
                np.column_stack([ts, tes, fit]),
                '%14.9f %9.3e %8.6f',
                header=header
            )

    if plot and ppath:

        # plot paths rel to star 3

        if model.model == 'triple' or model.model == 'tdisc':
            (x1s,y1s,z1s),(x2s,y2s,z2s),\
                (x3s,y3s,z3s) = model.paths(ts)
        elif model.model == 'quad2':
            (x1s,y1s,z1s),(x2s,y2s,z2s),\
                (x3s,y3s,z3s),(x4s,y4s,z4s) = model.paths(ts)
        else:
            print('model = {:s} not recognised'.format(model.model))

        # plot circles to show region over which dips occur
        r1 = seclipse.model.sol2au(model['r1'][0])
        r2 = seclipse.model.sol2au(model['r2'][0])
        if model.model == 'triple' or model.model == 'quad2':
            r3 = seclipse.model.sol2au(model['r3'][0])
        elif model.model == 'tdisc':
            r3 = seclipse.model.sol2au(model['rdisc'][0])
        if model.model == 'quad2':
            r4 = seclipse.model.sol2au(model['r4'][0])
        theta = np.linspace(0,2.*np.pi,200)
        xc, yc = np.cos(theta), np.sin(theta)

        # black for star 3 only, blue for star 1 + 3 
        # eclipses, green for 2 + 3 eclipses.

        if model.model == 'quad2':
            print('blue = 1, green = 2, black = 3, red = 4')
        else:
            print('blue = 1, green = 2, black = 3')

        fig = plt.figure()

        # first panel star 1 (and 4) rel to 3
        ax1 = fig.add_subplot(121)
        ax1.set_aspect('equal')
        if model.model == 'tdisc':
            pfac = np.cos(np.radians(model['idisc'][0]))
            x = r3*xc
            y = r3*pfac*yc
            OMEGAdisc = np.radians(model['OMEGAdisc'][0])
            coso = np.cos(OMEGAdisc)
            sino = np.sin(OMEGAdisc)
            xt = -sino*x-coso*y
            yt = +coso*x-sino*y
            ax1.plot(xt,yt,'k')
        else:
            ax1.plot(r3*xc,r3*yc,'k')
            ax1.plot((r1+r3)*xc,(r1+r3)*yc,'b--')

        x1s -= x3s
        y1s -= y3s
        ax1.plot(x1s,y1s,'b')
        if model.model != 'tdisc':
            rsq = x1s**2+y1s**2
            mins = np.r_[True, rsq[1:] < rsq[:-1]] & \
                   np.r_[rsq[:-1] < rsq[1:], True]
            ax1.plot(x1s[mins],y1s[mins],'.b')
            ax1.plot(0,0,'ok',zorder=1)

        if model.model == 'quad2':
            x4s -= x3s
            y4s -= y3s
            ax1.plot(x4s,y4s,'r')
            ax1.plot((r4+r3)*xc,(r4+r3)*yc,'r--')
            ax1.set_title('Star 1 (and 4) rel to 3')
        else:
            ax1.set_title('Star 1 rel to 3')

        ax1.set_xlabel('X [AU]')
        ax1.set_ylabel('Y [AU]')
        ax1.set_xlim(-3*r3,3*r3)
        ax1.set_ylim(-3*r3,3*r3)

        # second panel star 2 (and 4) rel to 3
        ax2 = fig.add_subplot(122)
        ax2.set_aspect('equal')
        if model.model == 'tdisc':
            ax2.plot(xt,yt,'k')
        else:
            ax2.plot(r3*xc,r3*yc,'k')
            ax2.plot((r2+r3)*xc,(r2+r3)*yc,'g--')

        if model.model == 'quad2':
            ax2.plot((r4+r3)*xc,(r4+r3)*yc,'r--')
            ax2.plot(x4s,y4s,'r')
            ax2.set_title('Star 2 (and 4) rel to 3')
        else:
            ax2.set_title('Star 2 rel to 3')

        x2s -= x3s
        y2s -= y3s
        ax2.plot(x2s,y2s,'g')
        if model.model != 'tdisc':
            rsq = x2s**2+y2s**2
            mins = np.r_[True, rsq[1:] < rsq[:-1]] & \
                   np.r_[rsq[:-1] < rsq[1:], True]
            ax2.plot(x2s[mins],y2s[mins],'.g')
            ax2.plot(0,0,'ok',zorder=1)

        ax2.set_xlabel('X [AU]')
        ax2.set_ylabel('Y [AU]')
        ax2.set_xlim(-3*r3,3*r3)
        ax2.set_ylim(-3*r3,3*r3)

        plt.show()

