#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import trm.subs.input as inp
from trm import seclipse, orbits, cline

def ppxy(args=None):
    """``ppxy model time1 time2 ntime``

    Script to compute and plot the eclipse of limb-darkened
    spheres. It either does so given an already existing data file as
    a template or on a regularly-spaced set of times. In the latter
    case it will also plot a representation of the paths of the
    spheres.

    This one projects onto the plane of the sky (x--y)

    Arguments::

      model : str
         the file with the model

      time1 : float
         start time    

      time1 : float
         end time

      ntime : int
         number of times
    """

    command, args = cline.script_args(args)

    with cline.Cline("SECLIPSE_ENV", ".seclipse", command, args) as cl:
                     
        # register parameters
        cl.register('model', cline.Cline.LOCAL, cline.Cline.PROMPT)
        cl.register('time1', cline.Cline.GLOBAL, cline.Cline.PROMPT)
        cl.register('time2', cline.Cline.GLOBAL, cline.Cline.PROMPT)
        cl.register('ntime', cline.Cline.GLOBAL, cline.Cline.PROMPT)

        # get them
        mod = cl.get_value(
            'model', 'light curve model',
            cline.Fname('lc', '.mod')
        )
        model = seclipse.model.Model(mod)
        time1 = cl.get_value('time1', 'start time', 0.)
        time2 = cl.get_value('time2', 'end time', 100.)
        ntime = cl.get_value('ntime', 'number of times', 2)

    ts = np.linspace(time1, time2, ntime)

    # plot path of star 2 rel to star 3

    if model.model == 'triple' or model.model == 'tdisc':
        (x1s,y1s,z1s),(x2s,y2s,z2s),\
            (x3s,y3s,z3s) = model.paths(ts)
    elif model.model == 'quad2':
        (x1s,y1s,z1s),(x2s,y2s,z2s),\
            (x3s,y3s,z3s),(x4s,y4s,z4s) = model.paths(ts)
    else:
        print(f'model = {model.model} not recognised')

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

    fig,ax = plt.subplots()

    # star 1
    ax.set_aspect('equal')
    if model.model == 'tdisc':
        pass
#        ax1.plot(xt,yt,'k')
    else:
        ax.plot(r3*xc,r3*yc,'k',label='r3')
        ax.plot((r3+r1)*xc,(r3+r1)*yc,'b--',label='r3+r1')
        ax.plot((r3+r2)*xc,(r3+r2)*yc,'c--',label='r3+r2')

    ax.set_title('Star 1 & 4 rel to 3')

    if model.model == 'quad2':
        x4s -= x3s
        y4s -= y3s
        ax.plot(x4s,y4s,'r',label='star 4')

    x1s -= x3s
    y1s -= y3s
    x2s -= x3s
    y2s -= y3s
    ax.plot(x1s,y1s,'b',label='star 1')
    ax.plot(x2s,y2s,'c',label='star 2')
    ax.legend()
    
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')

    plt.show()
