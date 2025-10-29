"""
The purpose of this file is to trajectory preprocessing for perturbers moving through the gaseous medium.
We want a general method of prescrbing any trajectory, capable of evenly resampling the trajectory over time,
and performing a very efficient interpolation.
"""

import Utilities as utls
import numpy as np
import math
from scipy.interpolate import interp1d


class Trajectory():
	def __init__(self, PreEvaluatedPoints, N_Sample_times, plot = False):

		"""
		The array PreEvaluatedPoints is an array of shape (n , 4) containing 
		the coordinates (t, x, y, z) for n timesteps (pre-calculated). This class perfoms an interpolation
		of this data and returns an evenly sampled domain of shape (N_Sample_times , 4). 
		"""

		self.PreEvaluatedPoints   = PreEvaluatedPoints
		self.N_Sample_times       = int(N_Sample_times)
		self.plot                 = plot  # Option to plot the trajectory
		self.Time_i              = self.PreEvaluatedPoints[ 0,0]
		self.Time_f              = self.PreEvaluatedPoints[-1,0]
		self.Time_n              = len(self.PreEvaluatedPoints[:,0])
		self.TimeRange           = self.Time_f - self.Time_i

		# Cached values
		self._SampledTime         = None
		self._SampledTrajectory   = None 
		self._SampledPosition     = None 
		self._SampledVelocity     = None
		self._SampledAcceleration = None
		self._MaxSpeed            = None
		self._CreateInterpolator  = None
		
		self.Plot

	@property
	def ImportedShape(self):
		"""
		To check if the imported data is of the correct shape. If not, an error will be raised
		"""
		Shape = np.shape(self.PreEvaluatedPoints)
		if Shape[1] != 4:
			raise IndexError('Input array has incorrect shape of %g rather than (%g, 4)' %(Shape, Shape[0]))
		else:
			T = self.PreEvaluatedPoints[:,0]
			X = self.PreEvaluatedPoints[:,1]
			Y = self.PreEvaluatedPoints[:,2]
			Z = self.PreEvaluatedPoints[:,3]

			if len(T) == len(X) == len(Y) == len(Z):
				pass
			else:
				raise IndexError('Requires input shape of (N , 4) corresponding to N timesteps and 4 coordinates (t, x, y, z)')
			
		return Shape

	@property
	def T_Array(self):
		"""
		An evenly spaced array from the beginning to the end of the trajectory
		with N_Sample_times timesteps
		"""
		if self._SampledTime is None:
			self._SampledTime = np.linspace(self.Time_i, self.Time_f, self.N_Sample_times, endpoint = True)

		return self._SampledTime

	@property
	def MaxSpeed(self):
		dt       = np.diff(self.PreEvaluatedPoints[:,0])[:, None]
		dx       = np.diff(self.PreEvaluatedPoints[:,1:], axis = 0)
		velocity = dx/dt

		if self._MaxSpeed == None:
			self._MaxSpeed = np.max(np.linalg.norm(velocity, axis = 1))

		return self._MaxSpeed

	@property
	def CreateInterpolators(self):
		if self._CreateInterpolator is None:
		    t = self.PreEvaluatedPoints[:, 0]
		    x = self.PreEvaluatedPoints[:, 1:]

		    # Compute velocity and acceleration
		    Velocity_Vector     = np.gradient(x, t, axis=0)
		    Acceleration_Vector = np.gradient(Velocity_Vector, t, axis=0)

		    # Build interpolators
		    self.PositionInterp      = interp1d(t, x, axis=0, fill_value='extrapolate', kind='linear')
		    self.VelocityInterp      = interp1d(t, Velocity_Vector, axis=0, fill_value='extrapolate', kind='linear')
		    self.AccelerationInterp  = interp1d(t, Acceleration_Vector, axis=0, fill_value='extrapolate', kind='linear')
		    self._CreateInterpolator = True

	@property
	def EvenlySampledTrajectory(self):
	    """
	    Computes and returns (position, velocity, acceleration)
	    on the evenly spaced time grid.
	    """
	    if self._SampledTrajectory is None:
	        self.CreateInterpolators

	        # Cache the full sampled arrays
	        self._SampledTrajectory = (
	            self.PositionInterp(self.T_Array),
	            self.VelocityInterp(self.T_Array),
	            self.AccelerationInterp(self.T_Array)
	        )

	    return self._SampledTrajectory



	@property
	def X_Array(self):
		if self._SampledPosition is None:
			x, _, _               = self.EvenlySampledTrajectory
			self._SampledPosition = np.ascontiguousarray(x, dtype=np.float64)

		return self._SampledPosition

	@property
	def V_Array(self):
		if self._SampledVelocity is None:
			x_arr       = self.X_Array
			t_arr       = self.T_Array
			v_arr       = np.empty_like(x_arr)
			v_arr[1:-1] = (x_arr[2:] - x_arr[:-2]) / (t_arr[2:] - t_arr[:-2])[:, None]
			v_arr[0]    = v_arr[1]
			v_arr[-1]   = v_arr[-2]
			self._SampledVelocity = np.ascontiguousarray(v_arr, dtype=np.float64)

		return self._SampledVelocity

	@property
	def A_Array(self):
		if self._SampledAcceleration is None:
			x_arr       = self.X_Array
			a_arr       = np.empty_like(x_arr)
			t_arr       = self.T_Array
			dt          = t_arr[1] - t_arr[0]
			a_arr[1:-1] = (x_arr[2:] - 2 * x_arr[1:-1] + x_arr[:-2]) / (dt ** 2)
			a_arr[0]    = a_arr[1]
			a_arr[-1]   = a_arr[-2]
			self._SampledAcceleration = np.ascontiguousarray(a_arr, dtype=np.float64)

		return self._SampledAcceleration



	def Position(self, t):
		self.CreateInterpolators
		return self.PositionInterp(t)

	def Velocity(self,t):
		self.CreateInterpolators
		return self.VelocityInterp(t)

	def Acceleration(self,t):
		self.CreateInterpolators
		return self.AccelerationInterp(t)


	@property
	def Plot(self):
		if self.plot:
			# Test that the interpolation is working			

			import matplotlib.pyplot as plt
			fig, ax = plt.subplots(2, 2, figsize=(8, 8))
			plt.subplots_adjust(hspace=0.3, wspace=0.3)

			ax[0,0].plot(self.X_Array[:,0], self.X_Array[:,1])
			ax[0,0].set_title('Trajectory')
			ax[0,0].set_xlabel(r'$x$')
			ax[0,0].set_xlabel(r'$y$')

			ax[0,1].set_title('Speed')
			ax[0,1].plot(self.T_Array, self.V_Array[:,0], label = r'$v_x$')
			ax[0,1].plot(self.T_Array, self.V_Array[:,1], label = r'$v_y$')
			ax[0,1].plot(self.T_Array, self.V_Array[:,2], label = r'$v_z$')
			ax[0,1].legend(loc='upper left')
			ax[0,1].axhline(y = self.MaxSpeed, linestyle = 'dashed')
			ax[0,1].set_xlabel('Time')

			ax[1,0].set_title('Acceleration')
			ax[1,0].plot(self.T_Array, self.A_Array[:,0], label = r'$a_x$')
			ax[1,0].plot(self.T_Array, self.A_Array[:,1], label = r'$a_y$')
			ax[1,0].plot(self.T_Array, self.A_Array[:,2], label = r'$a_z$')
			ax[1,0].axhline(y = self.MaxSpeed, linestyle = 'dashed')
			ax[1,0].legend(loc='upper left')
			ax[1,0].set_xlabel('Time')
		
			plt.show()




if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # (1) Prescribe the motion
    n, N                = int(1e4), int(1e4)
    TrajectoryTest      = np.zeros([n,4])
    TrajectoryTest[:,0] = np.linspace(0,4,n)
    TrajectoryTest[:,1] = np.sin(2*np.pi * TrajectoryTest[:,0]) 
    TrajectoryTest[:,2] = np.cos(2*np.pi * TrajectoryTest[:,0]) 


    IniTraj = Trajectory(TrajectoryTest, N, plot=True)
    start   = time.time()
    
    T     = np.linspace(IniTraj.Time_i, IniTraj.Time_f, 100000)
    start = time.time()
    pos   = [IniTraj.Position(t) for t in T]
    end   = time.time()

    print('Time taken',end-start)
    print('Time per position call (without vectorisation)', (end-start) / N)

    start = time.time()
    pos   = IniTraj.Position(T)
    end   = time.time()

    print('Time taken',end-start)
    print('Time per position call (with vectorisation)', (end-start) / N)











