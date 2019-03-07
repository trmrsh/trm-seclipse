#!/usr/bin/env python

import sys
import math
import time
import numpy as np
import trm.subs.input as inp
from trm import seclipse, orbits, subs
from trm.subs import Vec3

def cpaths(args=None):

    """computes the paths of limb-darkened spheres as generated during
    light curve modelling. It can either compute to match an already
    existing data file or on a fine array of times.

    """

    if args is None:
        args = sys.argv.copy()

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('output', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ldat',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('data',  inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('time1', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('time2', inp.Input.LOCAL, inp.Input.PROMPT)
    inpt.register('ntime', inp.Input.LOCAL, inp.Input.PROMPT)

    # get them
    mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
    model = seclipse.model.Model(mod)

    outfile = inpt.get_value('output', 'output synthetic data', subs.Fname('fake', '.paths', subs.Fname.NEW))

    ldat = inpt.get_value('ldat', 'load data file?', True)
    if ldat:
        dat  = inpt.get_value('data', 'light curve data', subs.Fname('lc', '.dat'))
        inpt.save()

        # Load just the times from a data file
        ts = np.loadtxt(dat,usecols=(0))

    else:
        time1 = inpt.get_value('time1', 'start time', 0.)
        time2 = inpt.get_value('time2', 'end time', 100.)
        ntime = inpt.get_value('ntime', 'number of times', 2)
        inpt.save()

        ts  = np.linspace(time1, time2, ntime)

    # compute the paths
    paths = model.paths(ts)

    if model.model == 'triple' or model == 'tdisc':
        (x1,y1,z1),(x2,y2,z2),(x3,y3,z3) = paths

        a1 = model['a1'][0]
        a2 = model['a2'][0]
        xb,yb,zb = (a2*x1+a1*x2)/(a1+a2),(a2*y1+a1*y2)/(a1+a2),(a2*z1+a1*z2)/(a1+a2)

        alist = [ts,x1,y1,z1,x2,y2,z2,x3,y3,z3,xb,yb,zb]

        header = """
This file contains x,y,z data representing the path of each of the components
of a triple star light curve model and was computed by cpaths.py starting from
the input model file {:s}.

There are 13 columns. The first contains the times in days, the next 12 come
in 4 x,y,z triplets representing the three stellar components in turn and the
centre of mass of the inner binary. The overall centre of mass is defined to
(0,0,0). The units are AU, z is towards Earth, x is West on the sky and y is
North. The units are AU.
""".format(mod)

    elif model.model == 'quad2':
        (x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4) = paths
        a1 = model['a1'][0]
        a2 = model['a2'][0]
        xb1,yb1,zb1 = (a2*x1+a1*x2)/(a1+a2),(a2*y1+a1*y2)/(a1+a2),(a2*z1+a1*z2)/(a1+a2)

        ab1 = model['ab1'][0]
        a4 = model['a4'][0]
        xb2,yb2,zb2 = (a4*xb1+ab1*x4)/(ab1+a4),(a4*yb1+ab1*y4)/(ab1+a4),(a4*zb1+ab1*z4)/(ab1+a4)

        alist = [ts,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,xb1,yb1,zb1,xb2,yb2,zb2]

        header = """
This file contains x,y,z data representing the path of each of the components
of a quad2 quadruple star light curve model and was computed by cpaths.py starting from
the input model file {:s}.

There are 19 columns. The first contains the times in days, the next 18 come in 6 x,y,z
triplets representing the four stellar components in turn, the centre of mass of the 
inner binary and finally the centre of mass of the intermediate binary. The overall
centre of mass is defined to (0,0,0). The units are AU, z is towards Earth, x is West
on the sky and y is North. The units are AU.
""".format(mod)

    else:
        print('model = {:s} not recogised'.format(model.model))
        exit(1)

    lcol = len(alist)
    fmt = ' '.join(['%.9e',] + (lcol-1)*['%.5e'])
    np.savetxt(outfile,np.column_stack(alist), fmt, header=header)
