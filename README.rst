seclipse is a module to compute the eclipse of multiple (3 or 4 at the moment)
limb-darkened spheres representing stars. It use trm.orbits to calculate the
positions which are represented by hierarchies of Keplerian 2-body orbits.

You will need to have installed two other modules of mine trm.subs and
trm.orbits, otherwise this should be a normal Python install. seclipse uses
Cython a fair amount to speed potentially lengthy calculations.

Tom Marsh
