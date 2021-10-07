trm.seclipse
============

trm.seclipse is a module to compute the eclipse of multiple (3 or 4 at
the moment) limb-darkened spheres representing stars. It uses
trm.orbits to calculate the positions which are represented by
hierarchies of Keplerian 2-body orbits.

You will need to have installed trm.vec3, trm.mcmc, trm.cline and
trm.orbits, otherwise this should be a normal Python install ('pip
install . --user'). seclipse uses Cython a fair amount to speed
potentially lengthy calculations.

Tom Marsh
