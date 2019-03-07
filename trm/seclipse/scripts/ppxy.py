#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import trm.subs.input as inp
from trm import seclipse, orbits, subs
from trm.subs import Vec3

def ppxy(args=None):
    """script to compute and plot the eclipse of limb-darkened spheres. It either
    does so given an already existing data file as a template or on a
    regularly-spaced set of times. In the latter case it will also plot a
    representation of the paths of the spheres.

    This one projects onto the plane of the sky (x--y)

    """

    # generate arguments
    if args is None:
        args = sys.argv.copy()

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('time1', inp.Input.GLOBAL, inp.Input.PROMPT)
    inpt.register('time2', inp.Input.GLOBAL, inp.Input.PROMPT)
    inpt.register('ntime', inp.Input.GLOBAL, inp.Input.PROMPT)

    # get them
    mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
    model = seclipse.model.Model(mod)

    time1 = inpt.get_value('time1', 'start time', 0.)
    time2 = inpt.get_value('time2', 'end time', 100.)
    ntime = inpt.get_value('ntime', 'number of times', 2)
    inpt.save()

    ts = np.linspace(time1, time2, ntime)

    # plot path of star 2 rel to star 3

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

    fig = plt.figure()

    # star 1
    ax1 = fig.add_subplot(211)
    ax1.set_aspect('equal')
    if model.model == 'tdisc':
        pass
#        ax1.plot(xt,yt,'k')
    else:
        ax1.plot(r3*xc,r3*yc,'k')
        ax1.plot((r1+r3)*xc,(r1+r3)*yc,'b--')

    ax1.set_title('Star 1 & 4 rel to 3')

    if model.model == 'quad2':
        x4s -= x3s
        y4s -= y3s
        ax1.plot(x4s,y4s,'r')

    x1s -= x3s
    y1s -= y3s
    ax1.plot(x1s,y1s,'b')

    ax1.set_xlabel('X [AU]')
    ax1.set_ylabel('Y [AU]')
#    ax1.set_ylim(-0.03,0.03)

    # star 2
    ax2 = fig.add_subplot(212)
    ax2.set_aspect('equal')
    if model.model == 'tdisc':
        pass
#        ax2.plot(xt,yt,'k')
    else:
        ax2.plot(r3*xc,r3*yc,'k')
        ax2.plot((r2+r3)*xc,(r2+r3)*yc,'g--')

    ax2.set_title('Star 2 rel to 3')

    if model.model == 'quad2':
        ax2.plot(x4s,y4s,'r')

    x2s -= x3s
    y2s -= y3s
    ax2.plot(x2s,y2s,'g')

    ax2.set_xlabel('X [AU]')
    ax2.set_ylabel('Y [AU]')
#    ax2.set_ylim(-0.03,0.03)

    plt.show()
