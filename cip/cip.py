#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cip.py
------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from .sod import Sod

class SodCIP(object):
  '''
  The density, pressure and specific internal energy are computed on a regular grid
  with two ghost cells on either end. The velocity is staggered on a half-index grid
  with one ghost cell on either end. Here's how the indices match up for an example
  with four "real" grid points and five "real" half-grid points:

  ```
                  G     G                             G     G
   REGULAR        0     1  |  2     3     4     5  |  6     7
   HALF              0  |  1     2     3     4     5  |  6
                     G                                   G

  ```
  
  '''
  
  def __init__(self, npoints = 200, dx = 0.01, dt = 0.001,
               gamma = 1.4, a = 0.65):
    '''
    
    '''
    
    # User
    self.npoints = npoints
    self.dx = dx
    self.dt = dt
    self.gamma = gamma
    self.a = a

    # Regular step bounds
    self.IBEG = 2
    self.IEND = self.IBEG + self.npoints
    self.ITOT = self.IEND + 2

    # Half step bounds
    self.HBEG = 1
    self.HEND = self.HBEG + 1 + self.npoints
    self.HTOT = self.HEND + 1
    
    # Initial profiles for the Sod shock tube
    self.rho = np.ones(self.ITOT); self.rho[self.ITOT//2:] = 0.125
    self.u = np.zeros(self.HTOT)
    self.p = np.ones(self.ITOT); self.p[self.ITOT//2:] = 0.1
    # NOTE: Division by rho in line below missing in Yabe & Aoki (1991)
    self.e = self.p / (self.gamma - 1) / self.rho 

    # Compute derivatives
    self.rhoprime = np.gradient(self.rho, self.dx)
    self.uprime = np.gradient(self.u, self.dx)
    self.eprime = np.gradient(self.e, self.dx)
    
  def InitPlot(self):
    '''
    
    '''
    
    # The x arrays
    self.x = np.arange(0, self.npoints) * self.dx
    self.xh = np.arange(0, (self.npoints + 1)) * self.dx - self.dx / 2
    self.xg = np.array([-2, -1, self.npoints, self.npoints + 1]) * self.dx
    self.xgh = np.array([-self.dx - self.dx / 2, (self.npoints + 1) * self.dx - self.dx / 2])
    
    # Plot initial state
    self.time = 0
    self.fig, self.ax = pl.subplots(2, 2, figsize = (14, 10), sharex = True)
    self.fig.subplots_adjust(hspace = 0.05)
    self.ax = self.ax.flatten()
    self.title = self.fig.suptitle('time = %.3f' % self.time, fontsize = 18)
    
    # Main grid
    self.curve_rho, = self.ax[0].plot(self.x, self.rho[self.IBEG:self.IEND], 'b-', label = 'CIP')
    self.curve_p, = self.ax[1].plot(self.x, self.p[self.IBEG:self.IEND], 'b-')
    self.curve_u, = self.ax[2].plot(self.xh, self.u[self.HBEG:self.HEND], 'b-')
    self.curve_e, = self.ax[3].plot(self.x, self.e[self.IBEG:self.IEND], 'b-')
        
    # Ghost cells
    self.ghost_rho, = self.ax[0].plot(self.xg, np.append(self.rho[:self.IBEG], self.rho[self.IEND:]), 'b.')
    self.ghost_p, = self.ax[1].plot(self.xg, np.append(self.p[:self.IBEG], self.p[self.IEND:]), 'b.')
    self.ghost_u, = self.ax[2].plot(self.xgh, np.append(self.u[:self.HBEG], self.u[self.HEND:]), 'b.')
    self.ghost_e, = self.ax[3].plot(self.xg, np.append(self.e[:self.IBEG], self.e[self.IEND:]), 'b.')
    
    # Analytic solution
    self.xa, self.rhoa, self.pa, self.ua, self.ea = Sod(self.time)
    self.analytic_rho, = self.ax[0].plot(self.xa, self.rhoa, 'r-', label = 'Analytic')
    self.analytic_p, = self.ax[1].plot(self.xa, self.pa, 'r-')
    self.analytic_u, = self.ax[2].plot(self.xa, self.ua, 'r-')
    self.analytic_e, = self.ax[3].plot(self.xa, self.ea, 'r-')
    
    # Labels
    self.ax[0].set_ylabel('Density', fontsize = 14)
    self.ax[1].set_ylabel('Pressure', fontsize = 14)
    self.ax[2].set_ylabel('Velocity', fontsize = 14)
    self.ax[3].set_ylabel('Energy', fontsize = 14)
    self.ax[2].set_xlabel('Distance', fontsize = 14)
    self.ax[3].set_xlabel('Distance', fontsize = 14)
    self.ax[0].legend(loc = 'upper right')
    
    # Limits
    self.SetLimits()

  def Animate(self, tend = 0.276, interval = 0, thin = 1):
    '''
    
    '''
    
    self.thin = thin
    self.tend = tend
    self.InitPlot()
    self.anim = animation.FuncAnimation(self.fig, self.UpdateFig, interval = interval)
    pl.show()
  
  def UpdateFig(self, *args):
    '''
    
    '''
    
    # Are we done yet?
    if self.time >= self.tend: 
      self.anim.event_source.stop()
      return
    
    # Take a step
    self.Step(self.thin)
    
    # Main curves
    self.title.set_text('time = %.3f' % self.time)
    self.curve_rho.set_ydata(self.rho[self.IBEG:self.IEND])
    self.curve_p.set_ydata(self.p[self.IBEG:self.IEND])
    self.curve_u.set_ydata(self.u[self.HBEG:self.HEND])
    self.curve_e.set_ydata(self.e[self.IBEG:self.IEND])
    
    # Ghost cells
    self.ghost_rho.set_ydata(np.append(self.rho[:self.IBEG], self.rho[self.IEND:]))
    self.ghost_p.set_ydata(np.append(self.p[:self.IBEG], self.p[self.IEND:]))
    self.ghost_u.set_ydata(np.append(self.u[:self.HBEG], self.u[self.HEND:]))
    self.ghost_e.set_ydata(np.append(self.e[:self.IBEG], self.e[self.IEND:]))
    
    # Analytic solution
    self.xa, self.rhoa, self.pa, self.ua, self.ea = Sod(self.time)
    self.analytic_rho.set_ydata(self.rhoa)
    self.analytic_p.set_ydata(self.pa)
    self.analytic_u.set_ydata(self.ua)
    self.analytic_e.set_ydata(self.ea)
    
    # Set plot limits
    self.SetLimits()
    
    return self.curve_rho, self.curve_p, self.curve_u, self.curve_e, \
           self.ghost_rho, self.ghost_p, self.ghost_u, self.ghost_e
  
  def SetLimits(self):
    '''
    
    '''
    
    self.ax[0].margins(0.05, None)
    
    for ax, var in zip([self.ax[0], self.ax[1], self.ax[2], self.ax[3]], 
                       [self.rhoa, self.pa, self.ua, self.ea]):
      vmin = var.min()
      vmax = var.max()
      vpad = max(0.2, 0.2 * (vmax - vmin))
      ax.set_ylim(vmin - vpad, vmax + vpad)
  
  def Step(self, nsteps = 1):
    '''
    
    '''
    
    for n in range(nsteps):
      
      # Non-advection phase: variables
      self.q = self.NumVisc()
      # NOTE: Multiplication by rho in line below missing in Yabe & Aoki (1991)
      self.p = (self.gamma - 1) * self.rho * self.e + self.q 
      self.rhostar = self.RhoStar()
      self.ustar = self.UStar()
      self.estar = self.EStar()

      # Non-advection phase: derivatives
      self.rhoprimestar = self.FPrimeStar(self.rho, self.rhostar, self.rhoprime)
      self.uprimestar = self.FPrimeStar(self.u, self.ustar, self.uprime, half = True)
      self.eprimestar = self.FPrimeStar(self.e, self.estar, self.eprime)

      # Advection phase
      self.rho, self.rhoprime = self.CIP0(self.rhostar, self.rhoprimestar)
      self.e, self.eprime = self.CIP0(self.estar, self.eprimestar)
      self.u, self.uprime = self.CIP0(self.ustar, self.uprimestar, half = True)
    
      # Advance
      self.time += self.dt
    
  def FPrimeStar(self, f, fstar, fprime, half = False):
    '''
    
    '''

    if not half:
      # NOTE: This differs from (19) in Yabe & Aoki (1991) by a factor of 2,
      # since we account for the fact that we're on a staggered half-step grid for u
      du = 2 * (self.u[self.HBEG+1:self.HEND] - self.u[self.HBEG:self.HEND-1])
      BEG = self.IBEG
      END = self.IEND
    else:
      du = (self.u[self.HBEG+1:self.HEND+1] - self.u[self.HBEG-1:self.HEND-1])
      BEG = self.HBEG
      END = self.HEND
    dfprimedt = np.zeros_like(f)
    dfprimedt[BEG:END] = (fstar[BEG+1:END+1] - fstar[BEG-1:END-1] - 
                         f[BEG+1:END+1] + f[BEG-1:END-1]) / (2 * self.dx * self.dt) - \
                         fprime[BEG:END] * du / (2 * self.dx)
    return fprime + self.dt * dfprimedt
  
  def RhoStar(self):
    '''
    
    '''
    
    drhodt = np.zeros_like(self.rho)
    drhodt[self.IBEG:self.IEND] = -(self.rho[self.IBEG:self.IEND] * 
                                   (self.u[self.HBEG+1:self.HEND] - 
                                   self.u[self.HBEG:self.HEND-1]) / self.dx)
    return self.rho + self.dt * drhodt

  def UStar(self):
    '''
    
    '''
    
    dudt = np.zeros_like(self.u)
    dudt[self.HBEG:self.HEND] = -((self.p[self.IBEG:self.IEND+1] - 
                                 self.p[self.IBEG-1:self.IEND]) / self.dx) / \
                                 (0.5 * (self.rho[self.IBEG:self.IEND+1] + 
                                 self.rho[self.IBEG-1:self.IEND]))
    return self.u + self.dt * dudt

  def EStar(self):
    '''
    
    '''
    
    dedt = np.zeros_like(self.p)
    dedt[self.IBEG:self.IEND] = -self.p[self.IBEG:self.IEND] / \
                                 self.rho[self.IBEG:self.IEND] * \
                                 (self.ustar[self.HBEG+1:self.HEND] - 
                                 self.ustar[self.HBEG:self.HEND-1] + 
                                 self.u[self.HBEG+1:self.HEND] - 
                                 self.u[self.HBEG:self.HEND-1]) / (2 * self.dx)
    return self.e + self.dt * dedt
  
  def CIP0(self, f, fprime, half = False):
    '''
    
    '''

    if not half:
      BEG = self.IBEG
      END = self.IEND
      uav = 0.5 * (self.ustar[self.HBEG+1:self.HEND] + self.ustar[self.HBEG:self.HEND-1])
      xi = -uav * self.dt
    else:
      BEG = self.HBEG
      END = self.HEND
      uav = self.u[self.HBEG:self.HEND]
      xi = -uav * self.dt

    # Negative velocity
    am = (fprime[BEG:END] + fprime[BEG+1:END+1]) / (self.dx ** 2) + \
          2 * (f[BEG:END] - f[BEG+1:END+1]) / (self.dx ** 3)
    bm = 3 * (f[BEG+1:END+1] - f[BEG:END]) / (self.dx ** 2) - \
        (2 * fprime[BEG:END] + fprime[BEG+1:END+1]) / self.dx
    
    # Positive velocity    
    ap = (fprime[BEG:END] + fprime[BEG-1:END-1]) / (self.dx ** 2) - \
          2 * (f[BEG:END] - f[BEG-1:END-1]) / (self.dx ** 3)
    bp = 3 * (f[BEG-1:END-1] - f[BEG:END]) / (self.dx ** 2) + \
        (2 * fprime[BEG:END] + fprime[BEG-1:END-1]) / self.dx
    
    a = np.where(uav < 0, am, ap)
    b = np.where(uav < 0, bm, bp)
    
    df = np.zeros_like(f)
    df[BEG:END] = ((a * xi + b) * xi + fprime[BEG:END]) * xi
    dfprime = np.zeros_like(f)
    dfprime[BEG:END] = (3 * a * xi + 2 * b) * xi
    fnext = f + df
    fprimenext = fprime + dfprime
    
    return fnext, fprimenext

  def NumVisc(self):
    '''
    
    '''

    visc = np.zeros_like(self.rho)
    du = (self.u[self.HBEG+1:self.HEND] - self.u[self.HBEG:self.HEND-1])
    c = np.sqrt(self.gamma * self.p / self.rho)[self.IBEG:self.IEND]
    visc[self.IBEG:self.IEND] =  np.where(du < 0,
                                  self.a * (-self.rho[self.IBEG:self.IEND] * c * du + 
                                  0.5 * (self.gamma + 1) * self.rho[self.IBEG:self.IEND] * du ** 2),
                                  0)
    if np.any(np.isnan(visc)):
      return np.zeros_like(self.p)
    else:
      return visc