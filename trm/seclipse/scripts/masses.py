#!/usr/bin/env python

from __future__ import division

import sys
import math
import time
import numpy as np
import trm.subs.input as inp
from trm import seclipse, orbits, subs
from trm.subs import Vec3
from trm.seclipse.model import sol2au, au2sol

def day2year(p):
    return subs.DAY/subs.YEAR*p

def masses(args=None):
    """``masses model``

    Prints out the masses of the components of a given model, triple, quadruple
    or whatever.
    """

    # generate arguments
    if args is None:
        args = sys.argv.copy()

    # generate arguments
    inpt = inp.Input('PYTHON_TRIPLE_ENV', '.pytriple', sys.argv)

    # register parameters
    inpt.register('model', inp.Input.LOCAL, inp.Input.PROMPT)

    # get them
    mod = inpt.get_value('model', 'light curve model', subs.Fname('lc', '.mod'))
    model = seclipse.model.Model(mod)
    inpt.save()

    if model.model == 'quad1':
        a1 = sol2au(model['a1'][0])
        a2 = sol2au(model['a2'][0])
        a3 = sol2au(model['a3'][0])
        a4 = sol2au(model['a4'][0])
        ab1 = sol2au(model['ab1'][0])
        ab2 = sol2au(model['ab2'][0])
        Pb1 = day2year(model['Pb1'][0])
        Pb2 = day2year(model['Pb2'][0])
        Pb3 = day2year(model['Pb3'][0])

        # Work out total masses of 1+2 and 3+4 from
        # binary 3
        m12 = ab2*(ab1+ab2)**2/Pb3**2
        m34 = ab1*(ab1+ab2)**2/Pb3**2

        m1 = a2*(a1+a2)**2/Pb1**2
        m2 = a1*(a1+a2)**2/Pb1**2

        m3 = a4*(a3+a4)**2/Pb2**2
        m4 = a3*(a3+a4)**2/Pb2**2

        print(
            'Binary 1: M1+M2 = {:6.3f}, M1 = {:6.3f}, M2 = {:6.3f}'.format(
                m1+m2,m1,m2)
        )

        print(
            'Binary 2: M3+M4 = {:6.3f}, M3 = {:6.3f}, M4 = {:6.3f}'.format(
                m3+m4,m3,m4)
        )

        print(
            'Binary 3: M1+M2 = {:6.3f}, M3+M4 = {:6.3f}'.format(
                m12,m34)
        )

    elif model.model == 'quad2':
        a1 = model['a1'][0]
        a2 = model['a2'][0]
        a3 = model['a3'][0]
        a4 = model['a4'][0]
        ab1 = model['ab1'][0]
        ab2 = model['ab2'][0]
        Pb1 = model['Pb1'][0]
        Pb2 = model['Pb2'][0]
        Pb3 = model['Pb3'][0]

        # work out combined 1+2 mass from binary 1
        m12 = ((a1+a2)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb1/subs.YEAR)**2
        m1 = a2/(a1+a2)*m12
        m2 = a1/(a1+a2)*m12

        print('Binary 1: M1+M2 = {:6.3f}, M1 = {:6.3f}, M2 = {:6.3f}'.format(
            m12,m1,m2))

        # Work out combined 1+2+4 mass from binary 2
        m124 = ((ab1+a4)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb2/subs.YEAR)**2
        M12 = a4/(ab1+a4)*m124
        m4 = ab1/(ab1+a4)*m124

        print('Binary 2: M1+M2+M4 = {:6.3f}, M1+M2 = {:6.3f}, M4 = {:6.3f}'.format(
            m124,M12,m4))

        # Work out combined 1+2+3+4 mass from binary 3
        m1234 = ((a3+ab2)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb3/subs.YEAR)**2
        M124 = a3/(a3+ab2)*m1234
        m3 = ab2/(a3+ab2)*m1234
        print('Binary 3: M1+M2+M3+M4 = {:6.3f}, M1+M2+M4 = {:6.3f}, M3 = {:6.3f}'.format(
            m1234,M124,m3))

        print('Kepler test 1: {:6.3f} vs {:6.3f} [M1+M2]'.format(m12,M12))
        print('Kepler test 2: {:6.3f} vs {:6.3f} [M1+M2+M4]'.format(m124,M124))

    elif model.model == 'triple':
        a1 = model['a1'][0]
        a2 = model['a2'][0]
        a3 = model['a3'][0]
        ab = model['ab'][0]
        Pb1 = model['Pb1'][0]
        Pb2 = model['Pb2'][0]

        # work out combined 1+2 mass from binary 1
        m12 = ((a1+a2)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb1/subs.YEAR)**2
        m1 = a2/(a1+a2)*m12
        m2 = a1/(a1+a2)*m12

        print('Binary 1: M1+M2 = {:6.3f}, M1 = {:6.3f}, M2 = {:6.3f}'.format(
            m12,m1,m2))

        # Work out combined 1+2+3 mass from binary 2
        m123 = ((ab+a3)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb2/subs.YEAR)**2
        M12 = a3/(ab+a3)*m123
        m3 = ab/(ab+a3)*m123

        print('Binary 2: M1+M2+M3 = {:6.3f}, M1+M2 = {:6.3f}, M3 = {:6.3f}'.format(
            m123,M12,m3))

        print('Kepler test: {:6.3f} vs {:6.3f} [M1+M2]'.format(m12,M12))

    elif model.model == 'tdisc':
        a1 = model['a1'][0]
        a2 = model['a2'][0]
        adisc = model['adisc'][0]
        ab = model['ab'][0]
        Pb1 = model['Pb1'][0]
        Pb2 = model['Pb2'][0]

        # work out combined 1+2 mass from binary 1
        m12 = ((a1+a2)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb1/subs.YEAR)**2
        m1 = a2/(a1+a2)*m12
        m2 = a1/(a1+a2)*m12

        print('Binary 1: M1+M2 = {:6.3f}, M1 = {:6.3f}, M2 = {:6.3f}'.format(
            m12,m1,m2))

        # Work out combined 1+2+3 mass from binary 2
        m123 = ((ab+adisc)*subs.RSUN/subs.AU)**3/(subs.DAY*Pb2/subs.YEAR)**2
        M12 = adisc/(ab+adisc)*m123
        m3 = ab/(ab+adisc)*m123

        print('Binary 2: M1+M2+M3 = {:6.3f}, M1+M2 = {:6.3f}, M3 = {:6.3f}'.format(
            m123,M12,m3))

        print('Kepler test: {:6.3f} vs {:6.3f} [M1+M2]'.format(m12,M12))

    else:
        print('Have not implemented model = {:s} yet'.format(model.model))
        exit(1)
