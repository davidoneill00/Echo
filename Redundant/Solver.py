"""
This file computes the density perturbations created by the perturbers prescribed in InitialiseTrajectory.py. 
"""
# import Solver.RootFinder as rf
# import numpy as np
# import Solver.Method_cpp
# from Solver.Method_cpp import bfs3d, UniqueArray_4d, Alpha3d


class DensityWakeSolver():
	def __init__(
		self, 
		Trajectory         : type, 
		sound_speed        : float, 
		Max_Number_of_Roots: int 
		):

		self.Trajectory          = Trajectory
		self.sound_speed         = sound_speed
		self.Max_Number_of_Roots = Max_Number_of_Roots

		self._t_arr = Trajectory.T_Array
		self._x_arr = Trajectory.X_Array
		self._v_arr = Trajectory.V_Array
		self._a_arr = Trajectory.A_Array

	def DistanceFunction(self, x1:float, y1:float, z1:float, x2:float, y2:float, z2:float) -> np.ndarray:
		dX = np.array([x2-x1 , y2-y1 , z2-z1])
		return np.sqrt(np.dot(dX,np.conjugate(dX)))

	def RetardedTime(self, ti:float, t:float, x:float, y:float, z:float) -> float:
		return t - self.DistanceFunction(*self.Trajectory.Position(ti), x, y, z) / self.sound_speed 

	def RootFunction(self, ti, t, x, y, z):
		"""
		At some field coordinate X1 = (t,x,y,z) we compute the distance to a point in the perturber's history (paramaterised by tr) 
		ie. X2 = (tr, x_p(tr), y_p(tr), z_p(tr)). The distance between X1 and X2 divided by the speed of sound (|X1-X2|/cs) can be used to 
		define a sound crossing time, allowing us to calculate how long ago that sound wave would have needed to be emitted from the old location
		(x_p(tr), y_p(tr), z_p(tr)). If the perturber happened to be at that exact location, at the previous time, then a sound wave
		was lanched to arrive exactly at the current coordinate, signifying the presence of a root and sourcing density perturbations.
		"""
		ti      = np.atleast_1d(ti)
		X_i     = self.Trajectory.Position(ti)     # shape (N, 3)
		x, y, z = np.broadcast_arrays(x, y, z)    # align shapes
		X       = np.stack([x, y, z], axis=-1)
		dX      = X_i - X                # broadcasted subtraction
		dist    = np.linalg.norm(dX, axis=-1)
		return ti - t + dist / self.sound_speed


	def RootValues(self, t:float, x:float, y:float, z:float, a:float, b:float, f_a:float, f_b:float) -> list:
		"""
		Finding the roots of 'RootFunction'. By defining the lower bound as the initial time over which the trajectory is parameterised
		and the upper bound as the current time, we ensure that (a) no roots exist from when before the perturber was interacting with
		the gas and (b) no acausal roots are created (ti>t).
		"""

		BrentTol            = 1e-12
		BrentIter           = 100
		Tol                 = 1e-5
		dx                  = 1e-4
		Offset              = 1e-3
		MaxIter             = 2000
		rhobeg              = 0.01
		Max_Number_of_Roots = self.Max_Number_of_Roots
		
		if a<b:
			RTS   = rf.GodTierRootFinder(self.RootFunction, BrentTol, BrentIter, Tol, dx, Offset, MaxIter, rhobeg, Max_Number_of_Roots, t, x, y, z).Roots(a,b,f_a,f_b)
			return RTS
			
		elif a == b:
			return []

		else:
			raise BoundaryError('Evaluation time is less than initial time (ie. t = %g should be less than ti = %g) --> Change the CustomTimeDomain' %(a, b))


	def IndividualRootContribution(self, tr, t, x, y, z):
		"""
		Given a root (tr), we calculate the density perturbation it produces at t, x, y, z.
		"""
		tr         = np.atleast_1d(tr)
		X_r        = self.Trajectory.Position(tr)
		V_r        = self.Trajectory.Velocity(tr)
		X_rel      = X_r - np.array([x, y, z])
		X_rel_norm = np.linalg.norm(X_rel, axis=1)
		RootWeight = 1.0 / np.abs(X_rel_norm + np.sum(X_rel * V_r, axis=1) / self.sound_speed)
		return RootWeight
	

	def Alpha(self, t, x, y, z):
		a, b     = float(self.Trajectory.Time_i), float(t)
		f_a, f_b = self.RootFunction([a, b], t, x, y, z)
		Roots    = self.RootValues(t, x, y, z, a, b, f_a[0], f_b[1])
		contrib  = self.IndividualRootContribution(Roots, t, x, y, z)
		return np.nansum(contrib)


	def QuickAlpha(self, t:float, x:float, y:float, z:float, roots:list):
		"""
		If the roots are already known, we can bypass the root-finding method to return the 
		density perturbations at a given time. 
		"""

		RootContribution = self.IndividualRootContribution(tr,roots,x,y,z)
		return np.sum(RootContribution)





class Walk:
	"""
	3D BFS-based solver for the linearised fluid wake in a cube.
	Starting from known root seeds (e.g. from DensityWakeSolver.RootValues), it 
	propagates through all topologically connected grid cells.
	"""

	def __init__(
		self, 
		Domain, 
		Trajectory, 
		Mach,
		Max_Number_of_Roots,
		print_status=False, 
		ndim=3
    	):

		self.Domain       = Domain
		self.ndim         = ndim
		self.Method       = DensityWakeSolver(Trajectory=Trajectory, sound_speed=Trajectory.MaxSpeed/Mach, Max_Number_of_Roots=Max_Number_of_Roots)
		self.print_status = print_status


	@property
	def InitialiseArrays(self):
		"""Allocate 3D arrays for roots and errors."""
		Nx, Ny, Nz = self.Domain.Resolution_x, self.Domain.Resolution_y, self.Domain.Resolution_z
		Nr         = self.Method.Max_Number_of_Roots
		Roots      = np.ascontiguousarray(np.full((Nx, Ny, Nz, Nr), np.nan), dtype=np.float64)
		Errors     = np.ascontiguousarray(np.full((Nx, Ny, Nz, Nr), np.nan), dtype=np.float64)
		return Roots, Errors

	def ComputeSeedRoots(self, t):
		origin_roots = []
		a, b         = self.Method.Trajectory.Time_i, t
		for origin in self.Domain.SeedOrigins:
			ix, iy, iz    = origin
			x, y, z       = self.Domain.X[ix], self.Domain.Y[iy], self.Domain.Z[iz]
			f_a, f_b      = self.Method.RootFunction([a, b], t, x, y, z)
			roots         = self.Method.RootValues(t, x, y, z, a, b, f_a, f_b)
			# Note: for multiple particle perturbers we do this for each one and keep track of each trajectory
			origin_roots.append((origin,roots))

		return origin_roots

	# ------------------------------------------------------------
	# ------------- Launch cpp compiled solver -------------------
	# ------------------------------------------------------------

	def launch_bfs(self, t, Roots, Errors, error_tol, unique_tol):
		"""
		Launch a Breadth First Search (BFS) algorithm from a corner to fill the entire domain
		"""

		if self.ndim != 3:
			raise ValueError("CoverCube requires ndim=3")

		Roots, Errors = bfs3d(
				Roots,
				Errors,
				self.Domain.X,
				self.Domain.Y,
				self.Domain.Z,
				np.ascontiguousarray(np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),dtype=np.float64),
				t,
				self.Method.sound_speed,
				error_tol,
				unique_tol,
				self.Method._t_arr,
				self.Method._x_arr,
				self.Method._v_arr,
				self.Method._a_arr
				)

		return Roots, Errors


	def single_wake(self, t, error_tol, unique_tol):
		"""
		BFS propagation across different roots.

		Parameters
		----------
		Domain : spatial grid object
		Method : DensityWakeSolver object
		t      : Evaluation time.

		"""

		# === 1. Compute the seeds and create empty containers ===
		origin_roots  = self.ComputeSeedRoots(t)
		Roots, Errors = self.InitialiseArrays

		# === 2. Compute seed roots using global solver ===		
		for (origin,roots) in origin_roots:
			ix, iy, iz           = origin
			Roots[ix, iy, iz, :] = roots

		Roots, Errors = self.launch_bfs(t, Roots, Errors, error_tol, unique_tol)
		# optional: save errors too

		# === 3. Use root grid to compute density perturbations ===
		GlobalAlpha, NRoots = Alpha3d(
			Roots,
			self.Domain.X,
			self.Domain.Y,
			self.Domain.Z,
			self.Method._t_arr,
			self.Method._x_arr,
			self.Method._v_arr,
			self.Method.sound_speed
			)

		print(f"[Single Wake] Global Alpha field assembled. Shape = {GlobalAlpha.shape}")
		return GlobalAlpha, NRoots, Roots
	
	


# This should combine both gravity and gas to provide an easy callable advance() function
# which exectutes the cpp kernels to advance time using Strang splitting. Careful to 
# update all classes and objects correctly. 



