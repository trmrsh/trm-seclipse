#!/usr/bin/env python

"""
plots surface brightness of disc in tilted disc model, first as an image
and then as a line plot
"""

from __future__ import division

import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import trm.subs.input as inp
from trm import seclipse, orbits, subs
from trm.subs import Vec3

if __name__ == '__main__':

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)

    # get them
    mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
    model = seclipse.model.Model(mod)
    inpt.save()
    if model.model != 'tdisc':
        print('this script only workd for tilted disc models (tdisc)')
        exit(1)

    sd1 = model['sdisc1'][0]
    sd2 = model['sdisc2'][0]
    sd3 = model['sdisc3'][0]
    sd4 = model['sdisc4'][0]
    sd5 = model['sdisc5'][0]
    sd6 = model['sdisc6'][0]
    rdisc = model['rdisc'][0]

    NSIDE = 1000
    img = np.zeros((NSIDE,NSIDE))
    x = np.linspace(-1,1,NSIDE)
    y = np.linspace(-1,1,NSIDE)
    X,Y = np.meshgrid(x,y)

    pfac = np.cos(np.radians(model['idisc'][0]))
    OMEGAdisc = np.radians(model['OMEGAdisc'][0])
    coso = np.cos(OMEGAdisc)
    sino = np.sin(OMEGAdisc)
    xt = -sino*X+coso*Y
    yt = (-coso*X-sino*Y)/pfac
    r = np.sqrt(xt**2+yt**2)
    rc = r[r<1]

    xmin, xmax = rdisc*X[r<1].min(), rdisc*X[r<1].max()
    ymin, ymax = rdisc*Y[r<1].min(), rdisc*Y[r<1].max()
    img[r<1] = sd1+rc*(sd2+rc*sd3+rc*(sd4+rc*(sd5+sd6*rc)))
    plt.imshow(img,origin='lower',vmin=0,aspect='equal',
               extent=(-rdisc,rdisc,-rdisc,rdisc))
    plt.colorbar()
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()

    x = np.linspace(0,1,100)
    y = sd1+x*(sd2+x*sd3+x*(sd4+x*(sd5+sd6*x)))
    plt.plot(x,y)
    plt.ylim(0,y.max())
    plt.show()
