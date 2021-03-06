#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import trm.subs.input as inp
from trm import seclipse, orbits, cline

def ppxz(args=None):

    """``ppxy model time1 time2 ntime``

    Script to plot paths of stars in x-z plane. x-z is close to the
    orbital plane if the Omegas are close to 270 or 90. +z points
    towards Earth so this plots -z so that the observer is best
    thought of as being at the bottom of the screen which is a more
    natural perspective.

    Arguments::

      model : str
         the file with the model

      time1 : float
         start time    

      time1 : float
         end time

      ntime : int
         number of times

      relative : bool
         whether to plot stars relative to star 3 or "absolute" (system
         centre of mass)

    """

    command, args = cline.script_args(args)

    with cline.Cline("SECLIPSE_ENV", ".seclipse", command, args) as cl:
                     
        # register parameters
        cl.register('model', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time1', cline.Cline.GLOBAL, cline.Cline.PROMPT)
        cl.register('time2', cline.Cline.GLOBAL, cline.Cline.PROMPT)
        cl.register('ntime', cline.Cline.GLOBAL, cline.Cline.PROMPT)
        cl.register('relative', cline.Cline.LOCAL, cline.Cline.PROMPT)

        # get them
        mod = cl.get_value(
            'model', 'light curve model',
            cline.Fname('lc', '.mod')
        )
        model = seclipse.model.Model(mod)
        time1 = cl.get_value('time1', 'start time', 0.)
        time2 = cl.get_value('time2', 'end time', 100.)
        ntime = cl.get_value('ntime', 'number of times', 2)
        relative = cl.get_value(
            'relative', 'paths relative to star 3 (or absolute)', True
        )
        
    ts = np.linspace(time1, time2, ntime)

    if model.model == 'triple' or model.model == 'tdisc':
        (x1s,y1s,z1s),(x2s,y2s,z2s),\
            (x3s,y3s,z3s) = model.paths(ts)
    elif model.model == 'quad2':
        (x1s,y1s,z1s),(x2s,y2s,z2s),\
            (x3s,y3s,z3s),(x4s,y4s,z4s) = model.paths(ts)
    else:
        print('model = {:s} not recognised'.format(model.model))

    if relative:
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

    fig,ax = plt.subplots()

    # star 1
    ax.set_aspect('equal')
    if relative:
        if model.model == 'tdisc':
            pass
        else:
            ax.plot(r3*xc,r3*yc,'k',label='r3')
            ax.plot((r3+r1)*xc,(r3+r1)*yc,'b--',label='r3+r1')
            ax.plot((r3+r2)*xc,(r3+r2)*yc,'c--',label='r3+r2')
        ax.set_title('Star 1, 2 and 4 relative to 3')
    else:
        ax.set_title('Star 1, 3 and 4')

    if model.model == 'quad2':
        if relative:
            x4s -= x3s
            z4s -= z3s
        ax.plot(x4s,-z4s,'r',label='star 4')

    if relative:
        x1s -= x3s
        z1s -= z3s
        x2s -= x3s
        z2s -= z3s
    else:
        ax.plot(x3s,-z3s,'g',label='star 3')

    if relative:
        ax.axvline(0,ls='--',color='k')
    else:
        ax.plot(0,0,'+k')
        
    ax.plot(x1s,-z1s,'b',label='star 1')
    ax.plot(x2s,-z2s,'c',label='star 2')
    ax.legend()
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('-Z [AU]')
    plt.show()
