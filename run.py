
from Domain import Cartesian3D
from Method import DensityWakeSolver
from Trajectory import Trajectory
from ParallelMPI import run_mpi
import numpy as np

# ------------- Initialise problem -------------

Mach                = 2
t                   = 5
n, N                = int(1e6), int(1e6)
TrajectoryTest      = np.zeros([n,4])
TrajectoryTest[:,0] = np.linspace(0,t,n)
TrajectoryTest[:,1] = np.sin(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,2] = np.cos(2*np.pi * TrajectoryTest[:,0]) 
TrajectoryTest[:,3] = np.zeros(n)

IniTraj             = Trajectory(TrajectoryTest, N, plot=False)
method              = DensityWakeSolver(IniTraj, sound_speed = IniTraj.MaxSpeed / Mach, Max_Number_of_Roots = 10)


Domain = Cartesian3D(rmin=0.01, SeedFraction = 0.02,
        Resolution_x=600, Min_x=-3, Max_x=3,
        Resolution_y=600, Min_y=-3, Max_y=3,
        Resolution_z=100, Min_z=0, Max_z=3)


# ------------- Run distributed computation -------------
results, nroots = run_mpi(Domain, method, t, error_tol=3e-7)



from mpi4py import MPI
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
if rank == 0 and results is not None and nroots is not None:
    import matplotlib.pyplot as plt
    plt.contourf(Domain.X, Domain.Y, np.log10(results[:,:,0]+1e-16), vmin = -1, vmax = 0.5, levels = 100, cmap = 'jet')
    plt.colorbar()
    plt.show()

    plt.contourf(Domain.X, Domain.Y, (nroots[:,:,0]), cmap='magma')
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


