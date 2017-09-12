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
dt = 0.00001
gamma = 1.4
a = 0.1
tend = 0.5
thin = 100

# Boundary conditions
boundary_rho = ('float', 'float')
boundary_u = ('float', 'float')
boundary_p = ('float', 'float')

# Plotting bounds
rholim = (0, 1.25)
ulim = (-0.25, 1.25)
plim = (0, 1.25)
elim = (1, 4)

# Radial grid
npoints = 200
rmin = 0.001
rmax = 2.
r = np.linspace(rmin, rmax, npoints)

# Initial profiles for the Sod shock tube in spherical coordinates
midpt = np.argmin(np.abs(r - (rmin + 0.5 * (rmax - rmin))))

rho = np.ones(npoints)
rho[:midpt] = 0.25
u = np.zeros(npoints + 1)
p = np.ones(npoints)
p[midpt:] = 0.71
p[:midpt] = 0.175

# Instantiate the solver
solver = Spherical(r, rho, u, p, dt = dt, gamma = gamma, a = a,
                   boundary_rho = boundary_rho, boundary_u = boundary_u, 
                   boundary_p = boundary_p, rholim = rholim, ulim = ulim,
                   plim = plim, elim = elim)

# Animate
solver.Animate(thin = thin, tend = tend)