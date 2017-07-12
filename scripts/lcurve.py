#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from trm import seclipse

s1     = 10.
s2     = 1. 
r1     = 0.025
r2     = 0.021
limb1  = seclipse.Limb(seclipse.Limb.POLY,0.6)
limb2  = seclipse.Limb(seclipse.Limb.POLY,0.3)
n1     = 50
n2     = 50
iangle = 89.
b1     = 0.001
b2     = 0.002

phase = np.linspace(-0.1,1.6,15000)
flux = seclipse.circle(phase,iangle,s1,s2,r1,r2,limb1,limb2,n1,n2,b1,b2)

plt.plot(phase, flux)
plt.show()
