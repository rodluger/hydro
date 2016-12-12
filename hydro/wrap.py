#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
wrap.py
-------

Ctypes wrapping routines for chydro.c

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import ctypes
import platform
import sys
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter

__all__ = ['Hydro']

# Constants
INT_LAXFRIED = 0
PI = 3.1415926535
MH = 1.673e-24
KBOLTZ = 1.381e-16
BIGG = 6.672e-8
MEARTH = 5.9742e27
REARTH = 6.3781e8
XUVEARTH = 4.64

class PLANET(ctypes.Structure):
  '''
  
  '''
  
  _fields_ = [("dMass", ctypes.c_double),
              ("dR0", ctypes.c_double),
              ("dT0", ctypes.c_double),
              ("dGamma", ctypes.c_double),
              ("dBeta", ctypes.c_double),
              ("dEpsXUV", ctypes.c_double),
              ("dSigXUV", ctypes.c_double),
              ("dFXUV", ctypes.c_double),
              ("dN0", ctypes.c_double),
              ("dQA", ctypes.c_double),
              ("dQB", ctypes.c_double)]
        
class SYSTEM(ctypes.Structure):
  '''
  
  '''
  
  _fields_ = [("iNGrid", ctypes.c_int),
              ("iNRuns", ctypes.c_int),
              ("iIntegrator", ctypes.c_int),
              ("iOutputTime", ctypes.c_int),
              ("dRMax", ctypes.c_double),
              ("dEps", ctypes.c_double),
              ("dTol", ctypes.c_double),
              ("dGridPower", ctypes.c_double)]

class OUTPUT(ctypes.Structure):
  '''
  
  '''
  
  _fields_ = [("iOutputNum", ctypes.c_int),
              ("iSize", ctypes.c_int),
              ("dR", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dV", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dT", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dRho", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dU0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dU1", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dU2", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dG0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dG1", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dG2", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dQ0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dQ1", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dQ2", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dMVisc", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dPVisc", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dEVisc", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
              ("dMDot", ctypes.c_double),
              ("dRXUV", ctypes.c_double)]

class Hydro(object):
  '''
  
  '''
  
  def __init__(self, **kwargs):
    '''
    
    '''
    
    # Init
    self._system = SYSTEM()
    self._planet = PLANET()
    self._output = OUTPUT()
    self._reset(**kwargs)
    
    # Load c module
    self._lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chydro.so'))  
    self._func = self._lib.fiHydro
    self._func.restype = ctypes.c_int
    self._func.argtypes = [SYSTEM, PLANET, ctypes.POINTER(OUTPUT)]
    
    self._dbl_free = self._lib.dbl_free
    self._dbl_free.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int]
    
  def _reset(self, **kwargs):
    '''
  
    '''
    
    # Planet params
    self.mass = kwargs.get('mass', 1.)
    self.r0 = kwargs.get('r0', 1.)
    self.t0 = kwargs.get('t0', 250.)
    self.gamma = kwargs.get('gamma', 5. / 3.)
    self.xuv_efficiency = kwargs.get('xuv_efficiency', 0.15)
    self.xuv_cross_section = kwargs.get('xuv_cross_section', 1.0e-18)
    self.xuv_flux = kwargs.get('xuv_flux', 100.)
    self.n0 = kwargs.get('n0', 5.0e12)
    
    # System params
    self.grid_points = kwargs.get('grid_points', 100)
    self.max_iter = kwargs.get('max_iter', 1e5)
    self.integrator = kwargs.get('integrator', 'Lax-Friedrichs')
    self.out_time = kwargs.get('out_time', 1e2)
    self.max_radius = kwargs.get('max_radius', 20.)
    self.int_epsilon = kwargs.get('int_epsilon', 0.5)
    self.int_tolerance = kwargs.get('int_tolerance', 1.e-10)
    self.grid_power = kwargs.get('grid_power', 1.0)
    
    # Internal
    self.converged = False
    self._executed = False
    
  def run(self):
    '''
    
    '''
    
    self.converged = bool(self._func(self._system, self._planet, self._output))
    self._executed = True
  
  @property
  def temperature(self):
    '''
    
    '''
    
    if self._executed:
      return self.t0 * np.array([[self._output.dT[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def radius(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dR[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def density(self):
    '''
    
    '''
    
    if self._executed:
      return self.n0 * np.array([[self._output.dRho[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def velocity(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dV[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def u0(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dU0[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def u1(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dU1[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def u2(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dU2[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def g0(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dG0[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def g1(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dG1[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def g2(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dG2[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def q0(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dQ0[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def q1(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dQ1[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def q2(self):
    '''
    
    '''
    
    return np.array([[self._output.dQ2[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])

  @property
  def mvisc(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dMVisc[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def pvisc(self):
    '''
    
    '''
    
    if self._executed:
      return np.array([[self._output.dPVisc[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def evisc(self):
    '''
    
    '''
    
    return np.array([[self._output.dEVisc[i][j] for j in range(self.grid_points)] for i in range(self._output.iOutputNum)])
  
  @property
  def mass(self):
    '''
    Planet mass in Earth masses. Default `1`.
    
    '''
    if self._executed:
      return self._planet.dMass / MEARTH
  
  @mass.setter
  def mass(self, v):
    self._planet.dMass = v * MEARTH
  
  @property
  def r0(self):
    '''
    Radius at lower boundary in Earth radii. Default `1`.
    
    '''
    return self._planet.dR0 / REARTH
  
  @r0.setter
  def r0(self, v):
    self._planet.dR0 = v * REARTH
  
  @property
  def t0(self):
    '''
    Temperature at the lower boundary. Default `250`.
    
    '''
    return self._planet.dT0
  
  @t0.setter
  def t0(self, v):
    self._planet.dT0 = v
  
  @property
  def gamma(self):
    '''
    Atmospheric adiabatic exponent. Default `5/3`.
    
    '''
    return self._planet.dGamma
  
  @gamma.setter
  def gamma(self, v):
    self._planet.dGamma = v
  
  @property
  def xuv_efficiency(self):
    '''
    XUV absorption efficiency. Default `0.15`.
    
    '''
    return self._planet.dEpsXUV
  
  @xuv_efficiency.setter
  def xuv_efficiency(self, v):
    self._planet.dEpsXUV = v
  
  @property
  def xuv_cross_section(self):
    '''
    Atomic XUV cross section in cm^2. Default `1e-18`.
    
    '''
    return self._planet.dSigXUV
  
  @xuv_cross_section.setter
  def xuv_cross_section(self, v):
    self._planet.dSigXUV = v
  
  @property
  def xuv_flux(self):
    '''
    Total XUV flux, scaled to present Earth. Default `1`.
    
    '''
    return self._planet.dFXUV / XUVEARTH
  
  @xuv_flux.setter
  def xuv_flux(self, v):
    self._planet.dFXUV = v * XUVEARTH
  
  @property
  def n0(self):
    '''
    Number density at the lower boundary. Default `5e12`.
    
    '''
    return self._planet.dN0
  
  @n0.setter
  def n0(self, v):
    self._planet.dN0 = v
  
  @property
  def grid_points(self):
    '''
    Number of vertical grid points. Default `100`.
    
    '''
    return self._system.iNGrid
  
  @grid_points.setter
  def grid_points(self, v):
    self._system.iNGrid = int(v)
  
  @property
  def max_iter(self):
    '''
    Maximum number of iterations. Default `1e5`.
    
    '''
    return self._system.iNRuns
  
  @max_iter.setter
  def max_iter(self, v):
    self._system.iNRuns = int(v)
  
  @property
  def integrator(self):
    '''
    Integration scheme. Default `Lax-Friedrichs`.
    
    '''
    if self._system.iIntegrator == INT_LAXFRIED:
      return 'Lax-Friedrichs'
    else:
      return 'Unknown'
  
  @integrator.setter
  def integrator(self, v):
    if v.lower() == 'lax-friedrichs':
      self._system.iIntegrator = INT_LAXFRIED
    else:
      raise ValueError('Currently, the only supported integrator is `Lax-Friedrichs`.')
  
  @property
  def out_time(self):
    '''
    Output every `out_time` iterations. Default `1e5`.
    
    '''
    return self._system.iOutputTime
  
  @out_time.setter
  def out_time(self, v):
    self._system.iOutputTime = int(v)
    
  @property
  def max_radius(self):
    '''
    Radius at upper boundary in Earth radii. Default `20`.
    
    '''
    return self._system.dRMax
  
  @max_radius.setter
  def max_radius(self, v):
    self._system.dRMax = v
  
  @property
  def int_epsilon(self):
    '''
    CFL condition constant. Default `0.5`.
    
    '''
    return self._system.dEps
  
  @int_epsilon.setter
  def int_epsilon(self, v):
    self._system.dEps = v
  
  @property
  def int_tolerance(self):
    '''
    Convergence tolerance. Default `1e-10`.
    
    '''
    return self._system.dTol
  
  @int_tolerance.setter
  def int_tolerance(self, v):
    self._system.dTol = v
  
  @property
  def grid_power(self):
    '''
    Radius grid power. Default `1`.
    
    '''
    return self._system.dGridPower
  
  @grid_power.setter
  def grid_power(self, v):
    self._system.dGridPower = v
  
  def free(self):
    '''
    Free the memory allocated for the C arrays.
    
    '''
    
    if self._executed:
      self._dbl_free(self._output.dR, self._output.iSize)
      self._dbl_free(self._output.dV, self._output.iSize)
      self._dbl_free(self._output.dT, self._output.iSize)
      self._dbl_free(self._output.dRho, self._output.iSize)
      self._dbl_free(self._output.dU0, self._output.iSize)
      self._dbl_free(self._output.dU1, self._output.iSize)
      self._dbl_free(self._output.dU2, self._output.iSize)
      self._dbl_free(self._output.dG0, self._output.iSize)
      self._dbl_free(self._output.dG1, self._output.iSize)
      self._dbl_free(self._output.dG2, self._output.iSize)
      self._dbl_free(self._output.dQ0, self._output.iSize)
      self._dbl_free(self._output.dQ1, self._output.iSize)
      self._dbl_free(self._output.dQ2, self._output.iSize)
      self._dbl_free(self._output.dMVisc, self._output.iSize)
      self._dbl_free(self._output.dPVisc, self._output.iSize)
      self._dbl_free(self._output.dEVisc, self._output.iSize)
  
  def __del__(self):
    '''
    Free the C arrays when the last reference to the class goes out of scope.
    
    '''
    
    self.free()
  
  def animate(self, interval = 50):
    '''
    
    '''
    
    if not self._executed:
      return None
    
    # The variables we will plot  
    vars = ['temperature', 'velocity', 'density', 'q0']
    labels = [r'$T\ (\mathrm{K})$', r'$v / c$', r'$\rho\ (\mathrm{g/cm^3})$', r'$Q\ (\mathrm{W/cm^3})$']
    yscale = ['linear', 'linear', 'log', 'linear']

    # Set up the animation
    fig, ax = pl.subplots(2, 2, figsize = (12, 8))
    fig.subplots_adjust(left = 0.1, right = 0.975, bottom = 0.1, top = 0.9, hspace = 0.3, wspace = 0.25)
    ax = ax.flatten()
    title = fig.suptitle('Iteration 0', fontsize = 18)
    
    # Plot the initial and final states
    curve = [None for var in vars]
    for i, var in enumerate(vars):
      ax[i].plot(self.radius[0], getattr(self, var)[0], 'r-', alpha = 0.2, label = 'Initial')
      ax[i].plot(self.radius[-1], getattr(self, var)[-1], 'b-', alpha = 0.2, label = 'Final')
      curve[i], = ax[i].plot(self.radius[0], getattr(self, var)[0], 'k-')
      ax[i].set_xlim(self.radius[0].min(), self.radius[-1].max())
      ax[i].set_ylabel(labels[i], fontsize = 16)
      ax[i].set_xlabel(r'$R\ (\mathrm{R_\oplus})$', fontsize = 16)
      ax[i].set_yscale(yscale[i])
      ax[i].set_xscale('log')
      ax[i].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].legend(loc = 'upper left', fontsize = 8)
    
    # Animate
    def updatefig(t):
      title.set_text('Iteration %d' % t)
      for i, var in enumerate(vars):
        curve[i].set_xdata(self.radius[t])
        curve[i].set_ydata(getattr(self, var)[t])
      return curve

    anim = animation.FuncAnimation(fig, updatefig, frames = self._output.iOutputNum, interval = interval, repeat = True)
    pl.show()
