from Setup.Domain import Cartesian3D
from Setup.Trajectory import Trajectory
from Solver.Method import DensityWakeSolver, Walk
import numpy as np
import matplotlib.pyplot as plt

class Setup():
    def __init__(
 		self,
        TrajectoryArray,
        sound_speed,
        Max_Number_of_Roots,
        rmin, 
        SeedFraction,
        Resolution_x, 
        Min_x, 
        Max_x,
        Resolution_y, 
        Min_y, 
        Max_y,
        Resolution_z, 
        Min_z, 
        Max_z,
        Nsample,
        error_tol,
        unique_tol
        ):

        # construct Domain, Trajectory and Method objects
        self.Trajectory = Trajectory(TrajectoryArray, N=Nsample, plot=False)
        self.Domain     = Cartesian3D(rmin=rmin, SeedFraction = SeedFraction,
                Resolution_x=Resolution_x, Min_x=Min_x, Max_x=Max_x,
                Resolution_y=Resolution_y, Min_y=Min_y, Max_y=Max_y,
                Resolution_z=Resolution_z, Min_z=Min_z, Max_z=Max_z
            )
        self.method     = Walk(self.Domain, self.Trajectory, Mach, Max_Number_of_Roots)
        

        # call method.single_wake()

        # save outputs

        # append timeseries

    



# ------------- Initialise problem -------------

Mach                = 5.0
Max_Number_of_Roots = 5
t                   = 3.0
n, N                = int(1e4), int(1e4)
TrajectoryTest      = np.zeros([n,4])
TrajectoryTest[:,0] = np.linspace(0,t,n)
# TrajectoryTest[:,1] = np.linspace(0,t,n) #np.sin(2*np.pi * TrajectoryTest[:,0]) 
# TrajectoryTest[:,2] = np.zeros(n)        #np.cos(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,1] = np.sin(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,2] = np.cos(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,3] = np.zeros(n)

IniTraj = Trajectory(TrajectoryTest, N, plot=False)
Domain  = Cartesian3D(rmin=0.01, SeedFraction = 0.01,
        Resolution_x=200, Min_x=-3, Max_x=3,
        Resolution_y=200, Min_y=-3, Max_y=3,
        Resolution_z=200, Min_z=0, Max_z=3)


method                 = Walk(Domain, IniTraj, Mach, Max_Number_of_Roots)
results, nroots, roots = method.single_wake(t, error_tol=3e-7, unique_tol=5e-4)


