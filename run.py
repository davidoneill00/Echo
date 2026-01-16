from Domain import Cartesian3D
from Method import DensityWakeSolver, Walk
from Trajectory import Trajectory
import numpy as np

# ------------- Initialise problem -------------

Mach                = 5.0
Max_Number_of_Roots = 5
t                   = 3.0
n, N                = int(1e4), int(1e4)
TrajectoryTest      = np.zeros([n,4])
TrajectoryTest[:,0] = np.linspace(0,t,n)
TrajectoryTest[:,1] = np.linspace(0,t,n) #np.sin(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,2] = np.zeros(n)        #np.cos(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,3] = np.zeros(n)

IniTraj = Trajectory(TrajectoryTest, N, plot=False)
Domain  = Cartesian3D(rmin=0.01, SeedFraction = 0.02,
        Resolution_x=200, Min_x=-3, Max_x=3,
        Resolution_y=200, Min_y=-3, Max_y=3,
        Resolution_z=200, Min_z=0, Max_z=3)


method                 = Walk(Domain, IniTraj, Mach, Max_Number_of_Roots)
results, nroots, roots = method.single_wake(t, error_tol=3e-7, unique_tol=5e-4)



import matplotlib.pyplot as plt

plt.contourf(Domain.X, Domain.Y, (nroots[:,:,0]), cmap='magma')
plt.colorbar()
plt.show()

plt.contourf(Domain.X, Domain.Y, np.log10(results[:,:,0]), cmap='jet', levels=np.linspace(-1,0.5,100))
plt.colorbar()
plt.show()

import sys
sys.exit()








from mpi4py import MPI
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()

if rank == 0 and results is not None:

    import pyvista as pv
    from pyvista import ImageData
    

    grid = ImageData()
    A    = np.log10(results)


    # Grid dimensions must be number_of_points, not cells
    grid.dimensions = np.array(A.shape) + 1  

    # Optional: set physical extents (recommended for real coordinates)
    grid.origin = (Domain.Extent_x[0], Domain.Extent_y[0], Domain.Extent_z[0])
    grid.spacing = (
        (Domain.Extent_x[1] - Domain.Extent_x[0]) / A.shape[0],
        (Domain.Extent_y[1] - Domain.Extent_y[0]) / A.shape[1],
        (Domain.Extent_z[1] - Domain.Extent_z[0]) / A.shape[2],
    )

    # Attach scalar field to grid cells
    grid.cell_data["Alpha"] = A.flatten(order="F")

    # --- Save for ParaView ---
    grid.save("Alpha_Global.vti")


