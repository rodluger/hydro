#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cip_spherical.py
----------------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
from cip import Spherical
import numpy as np

# Settings
dt = 0.0001
gamma = 1.4
a = 0.65
tend = 10
thin = 100

# Boundary conditions
boundary_rho = (1, 'float')
boundary_u = (0, 'float')
boundary_p = (1, 'float')

# Plotting bounds
rholim = (0, 1.25)
ulim = (-0.25, 1.25)
plim = (0, 1.25)
elim = (1, 4)

# Radial grid
npoints = 200
rmin = 1.
rmax = 5.
r = np.linspace(rmin, rmax, npoints)

# Initial profiles for the Sod shock tube in spherical coordinates
midpt = np.argmin(np.abs(r - (rmin + 0.5 * (rmax - rmin))))

print(midpt)

rho = np.ones(npoints)
rho[midpt:] = 0.125
u = np.zeros(npoints + 1)
p = np.ones(npoints)
p[midpt:] = 0.1

# Instantiate the solver
solver = Spherical(r, rho, u, p, dt = dt, gamma = gamma, a = a,
                   boundary_rho = boundary_rho, boundary_u = boundary_u, 
                   boundary_p = boundary_p, rholim = rholim, ulim = ulim,
                   plim = plim, elim = elim)

# Animate
solver.Animate(thin = thin, tend = tend)