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

BIGG = 6.67e-8
MPLANET = 5.974e27

class Spherical(object):
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
	
	def __init__(self, r, rho, u, p, dt = 0.001, gamma = 1.4, a = 0.65, CSL2 = False,
							 boundary_rho = ('float', 'float'), boundary_u = ('float', 'float'), 
							 boundary_p = ('float', 'float'), rholim = None, ulim = None, plim = None,
							 elim = None):
		'''
		
		:param r: The independent coordinate (length) array
		:param rho: The density array
		:param u: The velocity array
		:param p: The pressure array
		:param dt: The timestep (must be small enough to satisfy the CFL condition!)
		:param gamma: The adiabatic index of the gas
		:param a: A numerical viscosity tuning parameter
		:param CSL2: Use the CSL2 scheme for the density?
		:param boundary_rho: A tuple of boundary conditions (left and right) for the density. \
					 Each element should either be "float" or a value at which the density will be fixed.
		:param boundary_u: A tuple of boundary conditions (left and right) for the velocity. \
					 Each element should either be "float" or a value at which the density will be fixed.
		:param boundary_p: A tuple of boundary conditions (left and right) for the pressure. \
					 Each element should either be "float" or a value at which the density will be fixed.     
		:param rholim: Plotting limits for the density
		:param ulim: Plotting limits for the velocity
		:param plim: Plotting limits for the pressure
		:param elim: Plotting limits for the energy
		
		'''
		
		# User inputs
		self.dt = dt
		self.gamma = gamma
		self.a = a
		self.CSL2 = CSL2
		self.boundary_rho = boundary_rho
		self.boundary_u = boundary_u
		self.boundary_p = boundary_p
		self.rholim = rholim
		self.ulim = ulim
		self.plim = plim
		self.elim = elim
		
		# Initial mass & energy
		self.mass0 = 0
		self.energy0 = 0
		
		# Position, density, velocity, and pressure grids
		self.r = r
		self.rho = rho
		self.u = u
		self.p = p
		self.npoints = len(self.r)

		# Regular step bounds w/ two ghost cells
		self.IBEG = 2
		self.IEND = self.IBEG + self.npoints
		self.ITOT = self.IEND + 2

		# Half step bounds w/ one ghost cell
		self.HBEG = 1
		self.HEND = self.HBEG + 1 + self.npoints
		self.HTOT = self.HEND + 1
		
		# Add the ghost cells to the independent coordinate by linear extrapolation
		drlo = self.r[1] - self.r[0]
		drhi = self.r[-1] - self.r[-2]
		self.r = np.concatenate([[self.r[0] - 2 * drlo, self.r[0] - drlo], 
															self.r, 
														 [self.r[-1] + drhi, self.r[-1] + 2 * drhi]])
		
		# Compute the independent coordinate on the half-step (staggered) grid
		self.rh = 0.5 * (self.r[1:] + self.r[:-1])
		
		# Append the ghost cells to the dependent variables
		self.rho = np.concatenate([[self.rho[0], self.rho[0]], self.rho, [self.rho[-1], self.rho[-1]]])
		self.p = np.concatenate([[self.p[0], self.p[0]], self.p, [self.p[-1], self.p[-1]]])
		self.u = np.concatenate([[self.u[0]], self.u, [self.u[-1]]])

		# Compute the energy array
		self.e = self.p / (self.gamma - 1) / self.rho 

		# go back to this later, ignore for now
		u_i = np.concatenate([[self.u[0]],self.u,[self.u[-1]]])
		u_i = (u_i[:-1]+u_i[1:])/2.0

		# total energy from Kuramoto: including the kinetic energy term
		#self.e = self.rho*(u_i**2.0/2.0+self.p/(self.rho*(self.gamma-1.0))) # total energy

		# Compute dx array (forward difference, x[i+1] - x[i])
		self.dr = self.r[1:] - self.r[:-1]
		self.dr = np.append(self.dr, self.dr[-1])
		self.drh = self.rh[1:] - self.rh[:-1]
		self.drh = np.append(self.drh, self.drh[-1])

		# Set boundary conditions
		self.Boundary()
		
		# Compute derivatives
		self.rhoprime = np.gradient(self.rho, self.dr)
		self.uprime = np.gradient(self.u, self.drh)
		self.eprime = np.gradient(self.e, self.dr)
		
		# CSL2 not implemented right now
		# Compute initial density integral: Equation (19) in Yabe et al. (2001)
		if self.CSL2:
			self.eta = np.zeros(self.HTOT)
			self.eta[self.HBEG:self.HEND] = 0.5 * (self.rho[self.IBEG-1:self.IEND] + 
																						 self.rho[self.IBEG:self.IEND+1]) * \
																						 self.dr[self.IBEG-1:self.IEND]
			self.eta[0] = self.eta[1]
			self.eta[-1] = self.eta[-2]
	
	@property
	def energy(self):
		'''
		The total energy in the system
		
		'''
		
		return np.sum(self.e[self.IBEG:self.IEND])
	
	@property
	def mass(self):
		'''
		The total mass in the system
		
		'''
		
		return np.trapz(self.rho[self.IBEG:self.IEND], x = self.r[self.IBEG:self.IEND])
		 
	def InitPlot(self):
		'''
		
		'''
				
		# Plot initial state
		self.time = 0
		self.fig, self.ax = pl.subplots(2, 2, figsize = (12, 8), sharex = True)
		self.fig.subplots_adjust(hspace = 0.05)
		self.ax = self.ax.flatten()
		self.title = self.fig.suptitle('time = %.3f' % self.time, fontsize = 18)
		
		# Main grid
		self.curve_rho, = self.ax[0].plot(self.r[self.IBEG:self.IEND], self.rho[self.IBEG:self.IEND], 'b-', label = 'CIP')
		self.curve_p, = self.ax[1].plot(self.r[self.IBEG:self.IEND], self.p[self.IBEG:self.IEND], 'b-')
		self.curve_u, = self.ax[2].plot(self.rh[self.HBEG:self.HEND], self.u[self.HBEG:self.HEND], 'b-')
		self.curve_e, = self.ax[3].plot(self.r[self.IBEG:self.IEND], self.e[self.IBEG:self.IEND], 'b-')
				
		# Ghost cells
		self.ghost_rho, = self.ax[0].plot(np.append(self.r[:self.IBEG], self.r[self.IEND:]), np.append(self.rho[:self.IBEG], self.rho[self.IEND:]), 'b.')
		self.ghost_p, = self.ax[1].plot(np.append(self.r[:self.IBEG], self.r[self.IEND:]), np.append(self.p[:self.IBEG], self.p[self.IEND:]), 'b.')
		self.ghost_u, = self.ax[2].plot(np.append(self.rh[:self.HBEG], self.rh[self.HEND:]), np.append(self.u[:self.HBEG], self.u[self.HEND:]), 'b.')
		self.ghost_e, = self.ax[3].plot(np.append(self.r[:self.IBEG], self.r[self.IEND:]), np.append(self.e[:self.IBEG], self.e[self.IEND:]), 'b.')
				
		# Labels
		self.ax[0].set_ylabel('Density', fontsize = 14)
		self.ax[1].set_ylabel('Pressure', fontsize = 14)
		self.ax[2].set_ylabel('Velocity', fontsize = 14)
		self.ax[3].set_ylabel('Energy', fontsize = 14)
		self.ax[2].set_xlabel('Distance', fontsize = 14)
		self.ax[3].set_xlabel('Distance', fontsize = 14)
		self.ax[0].legend(loc = 'upper right')
		
		# Conserved quantities
		self.mass_label = self.ax[0].annotate('Mass error: %.5f' % 0, xy = (0.02,0.975), xycoords = 'axes fraction', ha = 'left', va = 'top')
		self.energy_label = self.ax[3].annotate('Energy error: %.5f' % 0, xy = (0.02,0.975), xycoords = 'axes fraction', ha = 'left', va = 'top')
		
		# Limits
		self.ax[0].set_ylim(self.rholim)
		self.ax[1].set_ylim(self.plim)
		self.ax[2].set_ylim(self.ulim)
		self.ax[3].set_ylim(self.elim)
		
	def Animate(self, tend = 0.276, interval = 1, thin = 1):
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
		
		# Initial quantities
		if self.mass0 == 0:
			self.mass0 = self.mass
			self.energy0 = self.energy
		
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
		
		# Conserved quantities
		self.mass_label.set_text('Mass error: %.5f' % ((self.mass - self.mass0) / self.mass0))
		self.energy_label.set_text('Energy error: %.5f' % ((self.energy - self.energy0) / self.energy0))
		
		# Limits
		self.ax[0].set_ylim(self.rholim)
		self.ax[1].set_ylim(self.plim)
		self.ax[2].set_ylim(self.ulim)
		self.ax[3].set_ylim(self.elim)
				
		return self.curve_rho, self.curve_p, self.curve_u, self.curve_e, \
					 self.ghost_rho, self.ghost_p, self.ghost_u, self.ghost_e
		
	def Step(self, nsteps = 1):
		'''
		
		'''
		
		for n in range(nsteps):
			
			# Set boundary conditions
			self.Boundary()
			
			# Non-advection phase: variables
			self.q = self.NumVisc()

			# NOTE: Multiplication by rho in line below missing in Yabe & Aoki (1991)
			self.p = (self.gamma - 1) * self.rho * self.e + self.q 

			# total energy, not specific internal energy
			# first, calculate u values on the regular grid

			u_i = np.concatenate([[self.u[0]],self.u,[self.u[-1]]])
			u_i = (u_i[:-1]+u_i[1:])/2.0

			# depending on what you use for energy, turned off for now
			#self.p = (self.e - self.rho*u_i**2.0/2.0)*(self.gamma-1.0)

			if not self.CSL2:
				self.rhostar = self.RhoStar()
			self.ustar = self.UStar()
			self.estar = self.EStar()

			# Non-advection phase: derivatives
			if not self.CSL2:
				self.rhoprimestar = self.FPrimeStar(self.rho, self.rhostar, self.rhoprime)
			self.uprimestar = self.FPrimeStar(self.u, self.ustar, self.uprime, half = True)
			self.eprimestar = self.FPrimeStar(self.e, self.estar, self.eprime)

			# Advection phase
			if not self.CSL2:
				self.rho, self.rhoprime = self.CIP0(self.rhostar, self.rhoprimestar)
			else:
				self.rho, self.eta = self.RhoCSL2()
			self.e, self.eprime = self.CIP0(self.estar, self.eprimestar)
			self.u, self.uprime = self.CIP0(self.ustar, self.uprimestar, half = True)
 			#self.e, self.eprime = self.CIP0(self.estar, self.eprimestar)
			# Advance
			self.time += self.dt
	
	def Boundary(self):
		'''
		
		'''
		
		# Left boundary: floating or fixed?
		if self.boundary_rho[0] == 'float':
				self.rho[0:self.IBEG] = self.rho[self.IBEG]
		else:
				self.rho[0:self.IBEG] = self.boundary_rho[0]
		
		if self.boundary_u[0] == 'float':
				self.u[0:self.HBEG] = 0
		else:
				self.u[0:self.HBEG] = self.boundary_u[0]
		
		if self.boundary_p[0] == 'float':
				self.p[0:self.IBEG] = self.p[self.IBEG]
		else:
				self.p[0:self.IBEG] = self.boundary_p[0]
		
		# Compute the energy    
		self.e[0:self.IBEG] = self.p[0] / (self.gamma - 1) / self.rho[0]
		
		# Right boundary: floating or fixed?
		if self.boundary_rho[1] == 'float':
				self.rho[self.IEND:] = self.rho[self.IEND-1]
		else:
				self.rho[self.IEND:] = self.boundary_rho[-1]
				
		if self.boundary_u[1] == 'float':
				self.u[self.HEND:] = 0
		else:
				self.u[self.HEND:] = self.boundary_u[-1]
				
		if self.boundary_p[1] == 'float':
				self.p[self.IEND:] = self.p[self.IEND-1]
		else:
				self.p[self.IEND:] = self.boundary_p[-1]
				
		# Compute the energy 
		self.e[self.IEND:] = self.p[self.IEND] / (self.gamma - 1) / self.rho[self.IEND]

		# total energy instead of specific internal energy
		u_i = np.concatenate([[self.u[0]],self.u,[self.u[-1]]])
		u_i = (u_i[:-1]+u_i[1:])/2.0


		#self.e[self.IEND:] = (self.e[self.IEND] - self.rho[self.IEND]*u_i[self.IEND]**2.0/2.0)*(self.gamma-1.0)
	def RhoCSL2(self):
		'''
		The notation is quite confusing here. Yabe et al. (2001) use `rho` to designate the
		integral of the variable of interest (`f`). However, since we're using CSL2 to solve for
		the time evolution of the density, I'm calling the integral of this quantity `eta`.
		
		'''
		raise NotImplementedError("Not yet implemented!")				
		# NOTE: It is unclear from Yabe et al. (2001) how xi should be calculated
		# NOTE: Might want to use `ustar` rather than `u`. Not sure. Either method seems to work.
		uav = 0.5 * (self.u[self.HBEG+1:self.HEND] + self.u[self.HBEG:self.HEND-1])
		xi = -uav * self.dt

		# Negative velocity: iup = i + 1; icell = i + 1/2
		dxm = self.x[self.IBEG+1:self.IEND+1] - self.x[self.IBEG:self.IEND]
		A1m = (self.rho[self.IBEG:self.IEND] + self.rho[self.IBEG+1:self.IEND+1]) / dxm ** 2 - \
					2 * self.eta[self.HBEG+1:self.HEND] / dxm ** 3
		A2m = -(2 * self.rho[self.IBEG:self.IEND] + self.rho[self.IBEG+1:self.IEND+1]) / dxm + \
					3 * self.eta[self.HBEG+1:self.HEND] / dxm ** 2
		
		# Positive velocity: iup = i - 1; icell = i - 1/2
		dxp = self.x[self.IBEG-1:self.IEND-1] - self.x[self.IBEG:self.IEND]
		A1p = (self.rho[self.IBEG:self.IEND] + self.rho[self.IBEG-1:self.IEND-1]) / dxp ** 2 + \
					2 * self.eta[self.HBEG:self.HEND-1] / dxp ** 3
		A2p = -(2 * self.rho[self.IBEG:self.IEND] + self.rho[self.IBEG-1:self.IEND-1]) / dxp - \
					3 * self.eta[self.HBEG:self.HEND-1] / dxp ** 2
		
		A1 = np.where(uav < 0, A1m, A1p)
		A2 = np.where(uav < 0, A2m, A2p)
		
		# Advection phase: One of the equations between (37) and (38) in Yabe et al. (2001)
		drho = np.zeros_like(self.rho)
		drho[self.IBEG:self.IEND] = 3 * A1 * xi ** 2 + 2 * A2 * xi
		rhostar = self.rho + drho
		
		# Nonadvection phase
		G = np.zeros_like(self.rho)
		# NOTE: Might want to use `ustar` rather than `u`. Not sure. Either method seems to work.
		G[self.IBEG:self.IEND] = -rhostar[self.IBEG:self.IEND] * \
														(self.u[self.HBEG+1:self.HEND] - 
														self.u[self.HBEG:self.HEND-1]) / self.dxh[self.HBEG:self.IEND-1]
		
		# Advance the density
		rhonext = rhostar + G * self.dt
		
		# Now worry about the integral of the density, `eta`. Note that Yabe et al. (2001)
		# confusingly call this `rho`. This is equation (28) in Yabe et al. (2001):
		deltaeta = np.zeros_like(self.eta)
		deltaeta[self.IBEG:self.IEND] = -(A1 * xi ** 3 + A2 * xi ** 2 + self.rho[self.IBEG:self.IEND] * xi)
		
		# NOTE: We need to calculate deltaeta in the innermost ghost cells since we're differentiating
		# it in the next step. So we'll need xi, A1 and A2 in the ghost cells. I *think* this is correct,
		# but this should be checked in the future.
		
		# Left ghost cell
		uav = 0.5 * (self.u[self.HBEG] + self.u[self.HBEG-1])
		xi = -uav * self.dt
		if uav < 0:
			dx = self.x[self.IBEG] - self.x[self.IBEG-1]
			A1 = (self.rho[self.IBEG-1] + self.rho[self.IBEG]) / dx ** 2 - 2 * self.eta[self.HBEG] / dx ** 3
			A2 = -(2 * self.rho[self.IBEG-1] + self.rho[self.IBEG]) / dx + 3 * self.eta[self.HBEG] / dx ** 2
		else:
			dx = self.x[self.IBEG-2] - self.x[self.IBEG-1]
			A1 = (self.rho[self.IBEG-1] + self.rho[self.IBEG-2]) / dx ** 2 + 2 * self.eta[self.HBEG-1] / dx ** 3
			A2 = -(2 * self.rho[self.IBEG-1] + self.rho[self.IBEG-2]) / dx - 3 * self.eta[self.HBEG-1] / dx ** 2
		deltaeta[self.IBEG-1] = -(A1 * xi ** 3 + A2 * xi ** 2 + self.rho[self.IBEG-1] * xi)
		
		# Right ghost cell
		uav = 0.5 * (self.u[self.HEND] + self.u[self.HEND-1])
		xi = -uav * self.dt
		if uav < 0:
			dx = self.x[self.IEND+1] - self.x[self.IEND]
			A1 = (self.rho[self.IEND] + self.rho[self.IEND+1]) / dx ** 2 - 2 * self.eta[self.HEND] / dx ** 3
			A2 = -(2 * self.rho[self.IEND] + self.rho[self.IEND+1]) / dx + 3 * self.eta[self.HEND] / dx ** 2
		else:
			dx = self.x[self.IEND-1] - self.x[self.IEND]
			A1 = (self.rho[self.IEND] + self.rho[self.IEND-1]) / dx ** 2 + 2 * self.eta[self.HEND-1] / dx ** 3
			A2 = -(2 * self.rho[self.IEND] + self.rho[self.IEND-1]) / dx - 3 * self.eta[self.HEND-1] / dx ** 2
		deltaeta[self.IEND] = -(A1 * xi ** 3 + A2 * xi ** 2 + self.rho[self.IEND] * xi)
		
		# Equation (38) in Yabe et al. (2001):
		deta = np.zeros_like(self.eta)
		deta[self.HBEG:self.HEND] = deltaeta[self.IBEG-1:self.IEND] - deltaeta[self.IBEG:self.IEND+1]
		etanext = self.eta + deta
		
		return rhonext, etanext
		
	def FPrimeStar(self, f, fstar, fprime, half = False):
		'''
		
		'''

		if not half:
			# NOTE: This differs from (19) in Yabe & Aoki (1991) by a factor of 2, since
			# we must account for the fact that we're on a staggered half-step grid for `u`
			du = 2 * (self.u[self.HBEG+1:self.HEND] - self.u[self.HBEG:self.HEND-1])
			BEG = self.IBEG
			END = self.IEND
		else:
			du = (self.u[self.HBEG+1:self.HEND+1] - self.u[self.HBEG-1:self.HEND-1])
			BEG = self.HBEG
			END = self.HEND
		dfprimedt = np.zeros_like(f)
		dfprimedt[BEG:END] = (fstar[BEG+1:END+1] - fstar[BEG-1:END-1] - 
												 f[BEG+1:END+1] + f[BEG-1:END-1]) / (2 * self.dr[BEG:END] * self.dt) - \
												 fprime[BEG:END] * du / (2 * self.dr[BEG:END])
		return fprime + self.dt * dfprimedt
	
	def RhoStar(self):
		'''
		
		'''
		
		drhodt = np.zeros_like(self.rho)
		ibeg = self.IBEG
		iend = self.IEND
		hbeg = self.HBEG
		hend = self.HEND

		term1 = -self.rho[ibeg:iend]/self.r[ibeg:iend]**2.0
		term2 = self.u[hbeg+1:hend]*self.rh[hbeg+1:hend]**2.0 - self.u[hbeg:hend-1]*self.rh[hbeg:hend-1]**2.0
		term3 = self.drh[hbeg:hend-1]

		drhodt[ibeg:iend] = term1*term2/term3

		'''
		drhodt[self.IBEG:self.IEND] = -(self.rho[self.IBEG:self.IEND] / \
																	 self.x[self.IBEG:self.IEND]**2 * \
																	 (self.u[self.HBEG+1:self.HEND] * self.xh[self.HBEG+1:self.HEND]**2.0 - \
																	 self.u[self.HBEG:self.HEND-1] * self.xh[self.HBEG:self.HEND-1]**2.0) / \
																	 self.dxh[self.HBEG:self.HEND-1])
		'''

		return self.rho + self.dt * drhodt

	def UStar(self):
		'''
		
		'''
		ibeg = self.IBEG
		iend = self.IEND
		hbeg = self.HBEG
		hend = self.HEND

		# bounds here don't match with Rodrigo's code: check this
		dudt = np.zeros_like(self.u)
		term1 = -2.0/(self.rho[ibeg-1:iend]+self.rho[ibeg:iend+1])*(self.p[ibeg:iend+1]-self.p[ibeg-1:iend])/self.dr[ibeg-1:iend]
		term2 = -BIGG*MPLANET/self.rh[hbeg:hend]
		# ignore term2 for now
		dudt[hbeg:hend] = term1# + term2

		'''
		dudt[self.HBEG:self.HEND] = -((self.p[self.IBEG:self.IEND+1] - 
																 self.p[self.IBEG-1:self.IEND]) / self.dx[self.IBEG:self.IEND+1]) / \
																 (0.5 * (self.rho[self.IBEG:self.IEND+1] + 
																 self.rho[self.IBEG-1:self.IEND]))
		'''
		return self.u + self.dt * dudt

	def EStar(self):
		'''
		
		ibeg = self.IBEG
		iend = self.IEND
		hbeg = self.HBEG
		hend = self.HEND

		dedt = np.zeros_like(self.p)

		u_i = np.concatenate([[self.u[0]],self.u,[self.u[-1]]])
		u_i = (u_i[:-1]+u_i[1:])/2.0

		term1 = -1.0/self.r[ibeg:iend]**2.0
		term2 = (self.u[hbeg+1:hend]*self.rh[hbeg+1:hend]**2.0 - self.u[hbeg:hend-1]*self.rh[hbeg:hend-1]**2.0)/self.drh[hbeg:hend-1]
		term3 = self.rho[ibeg:iend]*u_i[ibeg:iend]**2.0/2.0+self.p[ibeg:iend]*self.gamma/(self.gamma-1.0)
		term4 = -u_i[ibeg:iend]*(self.p[ibeg+1:iend+1]-self.p[ibeg:iend])/self.dr[ibeg:iend]
		term5 = -self.rho[ibeg:iend]*u_i[ibeg:iend]*BIGG*MPLANET/self.r[ibeg:iend]
		'''
		# heating term: q
		# conduction term with kappa and temperature gradient add later
		#dedt[ibeg:iend] = term1*term2*term3*term4#*term5
		'''
		dedt[self.IBEG:self.IEND] = -self.p[self.IBEG:self.IEND]/ \
																(self.rho[self.IBEG:self.IEND]*self.x[self.IBEG:self.IEND]**2) * \
																((self.ustar[self.HBEG+1:self.HEND]+self.u[self.HBEG+1:self.HEND]) *
																self.xh[self.HBEG+1:self.HEND]**2 -
																(self.ustar[self.HBEG:self.HEND-1]+self.u[self.HBEG:self.HEND-1]) *
																self.xh[self.HBEG:self.HEND-1]**2) / \
																(2.0*(self.xh[self.HBEG+1:self.HEND]-self.xh[self.HBEG:self.HEND-1]))
		'''
		dedt = np.zeros_like(self.p)
	
		# Pressure and density
		p = self.p[self.IBEG:self.IEND]
		rho = self.rho[self.IBEG:self.IEND]
		
		# Radius
		r = self.r[self.IBEG:self.IEND]
		
		# Velocity* above and below
		ustarp = self.ustar[self.HBEG+1:self.HEND]
		ustarm = self.ustar[self.HBEG:self.HEND-1]
		
		# Velocity above and below
		up = self.u[self.HBEG+1:self.HEND]
		um = self.u[self.HBEG:self.HEND-1]
		
		# Radius above and below (half step)
		rhp = self.rh[self.HBEG+1:self.HEND]
		rhm = self.rh[self.HBEG:self.HEND-1]
		
		# Delta r (half step)
		drh = self.drh[self.HBEG:self.HEND-1]
		
		# This is the third term in the G vector at the bottom of page 229
		dedt[self.IBEG:self.IEND] = -(p / (rho * r ** 2)) * (rhp ** 2 * (ustarp + up) - rhm ** 2 * (ustarm + um)) / (2 * drh)
		
		return self.e + self.dt * dedt

	
	def CIP0(self, f, fprime, half = False):
		'''
		
		'''

		if not half:
			BEG = self.IBEG
			END = self.IEND
			uav = 0.5 * (self.ustar[self.HBEG+1:self.HEND] + self.ustar[self.HBEG:self.HEND-1])
			xi = -uav * self.dt
			dxm = self.r[self.IBEG+1:self.IEND+1] - self.r[self.IBEG:self.IEND]
			dxp = self.r[self.IBEG-1:self.IEND-1] - self.r[self.IBEG:self.IEND]
		else:
			BEG = self.HBEG
			END = self.HEND
			uav = self.u[self.HBEG:self.HEND]
			xi = -uav * self.dt
			dxm = self.r[self.HBEG+1:self.HEND+1] - self.r[self.HBEG:self.HEND]
			dxp = self.r[self.HBEG-1:self.HEND-1] - self.r[self.HBEG:self.HEND]

		# Negative velocity
		am = (fprime[BEG:END] + fprime[BEG+1:END+1]) / (dxm ** 2) + \
					2 * (f[BEG:END] - f[BEG+1:END+1]) / (dxm ** 3)
		bm = 3 * (f[BEG+1:END+1] - f[BEG:END]) / (dxm ** 2) - \
				(2 * fprime[BEG:END] + fprime[BEG+1:END+1]) / dxm
		
		# Positive velocity
		ap = (fprime[BEG:END] + fprime[BEG-1:END-1]) / (dxp ** 2) + \
					2 * (f[BEG:END] - f[BEG-1:END-1]) / (dxp ** 3)
		bp = 3 * (f[BEG-1:END-1] - f[BEG:END]) / (dxp ** 2) - \
				(2 * fprime[BEG:END] + fprime[BEG-1:END-1]) / dxp

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
		du = (self.u[self.HBEG+1:self.HEND]*self.rh[self.HBEG+1:self.HEND]**2.0 - \
		self.u[self.HBEG:self.HEND-1]*self.rh[self.HBEG:self.HEND-1]**2.0)
		c = np.sqrt(self.gamma * self.p / self.rho)[self.IBEG:self.IEND]
		visc[self.IBEG:self.IEND] =  np.where(du < 0,
																	self.a * (-self.rho[self.IBEG:self.IEND] * c * du + 
																	0.5 * (self.gamma + 1) * self.rho[self.IBEG:self.IEND] * du ** 2),
																	0)
		if np.any(np.isnan(visc)):
			return np.zeros_like(self.p)
		else:
			return visc