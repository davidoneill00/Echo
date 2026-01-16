"""
The purpose of this file is to initialise both the spatial (and soon temporal) domain over which the density wake is evaluated.
"""
import numpy as np
import math 

class Cartesian3D():
	def __init__(self, 
		rmin, SeedFraction,
		Resolution_x, Min_x, Max_x,
		Resolution_y, Min_y, Max_y,
		Resolution_z, Min_z, Max_z,

		):

		self.rmin           = rmin
		self.Resolution_x   = Resolution_x
		self.Resolution_y   = Resolution_y
		self.Resolution_z   = Resolution_z
		self.Extent_x       = (Min_x, Max_x)
		self.Extent_y       = (Min_y, Max_y)
		self.Extent_z       = (Min_z, Max_z)
		self.SeedFraction   = SeedFraction 
		self.Cartesian      = True

		# cache value
		self._SeedOrigins   = None
		self._increment     = int(1/self.SeedFraction)
 
	@property
	def X(self):
		return np.linspace(self.Extent_x[0] ,self.Extent_x[1] , self.Resolution_x)

	@property
	def Y(self):
		return np.linspace(self.Extent_y[0] ,self.Extent_y[1] , self.Resolution_y)

	@property
	def Z(self):
		return np.linspace(self.Extent_z[0] ,self.Extent_z[1] , self.Resolution_z)

	@property
	def Seed_X(self):
		return self.X[0::self._increment]

	@property
	def Seed_Y(self):
		return self.Y[0::self._increment]

	@property
	def Seed_Z(self):
		return self.Z[0::self._increment]
	
	@property
	def SeedOrigins(self):
		if self._SeedOrigins == None:
			lst = []
			k   = 0
			while k <= self.Resolution_z - 1:
				j = 0
				while j <= self.Resolution_y - 1:
					i = 0
					while i <= self.Resolution_x - 1:
						lst.append((int(i), int(j), int(k)))
						i += self._increment
					j += self._increment
				k += self._increment

			self._SeedOrigins = lst

		return self._SeedOrigins
		


class CartesianLog3D():
	def __init__(self, 
		rmin, SeedFraction,
		Resolution_x, Min_x, Max_x,
		Resolution_y, Min_y, Max_y,
		Resolution_z, Min_z, Max_z,
		):

		self.rmin           = rmin
		self.Resolution_x   = Resolution_x
		self.Resolution_y   = Resolution_y
		self.Resolution_z   = Resolution_z
		self.Extent_x       = (Min_x, Max_x)
		self.Extent_y       = (Min_y, Max_y)
		self.Extent_z       = (Min_z, Max_z)
		self.SeedFraction   = SeedFraction 
		self.Cartesian      = False

		# cache value
		self._SeedOrigins   = None
		self._increment     = int(1/self.SeedFraction)
 
	@property
	def Midpoint(self):
		return [
			(self.Extent_x[0] + self.Extent_x[1]) / 2,
			(self.Extent_y[0] + self.Extent_y[1]) / 2,
			(self.Extent_z[0] + self.Extent_z[1]) / 2
		]

	@property
	def X(self):
		dx = (self.Extent_x[1] - self.Extent_x[0])
		xr = np.logspace(np.log10(self.rmin), np.log10(dx/2), self.Resolution_x//2)
		xl = -xr
		return np.array(xl.tolist() + xr.tolist()) + self.Midpoint[0]
	
	@property
	def Y(self):
		dy = (self.Extent_y[1] - self.Extent_y[0])
		yr = np.logspace(np.log10(self.rmin), np.log10(dy/2), self.Resolution_y//2)
		yl = -yr
		return np.array(yl.tolist() + yr.tolist()) + self.Midpoint[1]

	@property
	def Z(self):
		dz = (self.Extent_z[1] - self.Extent_z[0])
		zr = np.logspace(np.log10(self.rmin), np.log10(dz/2), self.Resolution_z//2)
		zl = -zr
		return np.array(zl.tolist() + zr.tolist()) + self.Midpoint[2]
	
	@property
	def Seed_X(self):
		return self.X[0::self._increment]

	@property
	def Seed_Y(self):
		return self.Y[0::self._increment]

	@property
	def Seed_Z(self):
		return self.Z[0::self._increment]
	
	@property
	def SeedOrigins(self):
		if self._SeedOrigins == None:
			lst = []
			k   = 0
			while k <= self.Resolution_z - 1:
				j = 0
				while j <= self.Resolution_y - 1:
					i = 0
					while i <= self.Resolution_x - 1:
						lst.append((int(i), int(j), int(k)))
						i += self._increment
					j += self._increment
				k += self._increment

			self._SeedOrigins = lst

		return self._SeedOrigins

















