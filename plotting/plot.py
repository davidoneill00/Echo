import h5py
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
import pyvista as pv
from scipy.ndimage import gaussian_filter

# -----------------------------
# Containers (simple namespaces)
# -----------------------------
class _Domain: pass
class _Trajectory: pass
class _Params: pass

Domain     = _Domain()
Trajectory = _Trajectory()
Params     = _Params()

# ---------------------------------------------
# Load HDF5 files into python readable format
# ---------------------------------------------
def load(filename):
    global Domain, Trajectory, Alpha, Params

    with h5py.File(filename, "r") as f:

        # -------------------------
        # Fixed params / domain
        # -------------------------
        Params.seed_fraction = f["/params/fixed/seed_fraction"][()]
        Params.max_number_of_roots = f["/params/fixed/max_number_of_roots"][()]

        Domain.Nx = int(f["/params/fixed/domain/Nx"][0])
        Domain.Ny = int(f["/params/fixed/domain/Ny"][0])
        Domain.Nz = int(f["/params/fixed/domain/Nz"][0])

        Domain.range_x = f["/params/fixed/domain/range_x"][:]
        Domain.range_y = f["/params/fixed/domain/range_y"][:]
        Domain.range_z = f["/params/fixed/domain/range_z"][:]

        # Construct coordinates (optional but useful)
        Domain.x = np.linspace(*Domain.range_x, Domain.Nx)
        Domain.y = np.linspace(*Domain.range_y, Domain.Ny)
        Domain.z = np.linspace(*Domain.range_z, Domain.Nz)

        # -------------------------
        # Trajectory
        # -------------------------
        Trajectory.t = f["/trajectory/t"][:]
        Trajectory.M = f["/trajectory/M"][:]
        Trajectory.X = f["/trajectory/X"][:]   # (N,3)
        Trajectory.V = f["/trajectory/V"][:]   # (N,3)
        Trajectory.A = f["/trajectory/A"][:]   # (N,3)

        Trajectory.iteration = f["/trajectory/iteration"][()]
        Trajectory.cadence   = f["/trajectory/RecordTrajectoryCadence_used"][()]
        Trajectory.Force     = f["/force/F"][()]
        Trajectory.Force_t   = f["/force/t"][()]

        Alpha   = f["/alpha/data"][:]   # (Nx,Ny,Nz)
        Alpha_t = f["/alpha/meta/t"][()]

    
def Ostriker99(v,c,t,rmin,tmin, GM=1, rho0 = 1):
    """
    Return the drag force as predicted by Ostriker (1999) for rectilinear motion for a given Mach number, velocity and time.
    """
    F  = 4 * np.pi * GM**2 * rho0 / v**2
    x1 = 0.97
    x2 = 1.02
    M  = v/c
    if M<=x1:
        return [-F * (0.5*math.log((1+M)/(1-M))-M),0]
    elif M>=x2:
        Lg   = math.log(v*(t-tmin)/rmin)
        return [-F * (0.5*math.log(1-1/(M**2))+Lg),0]
    else:
        c1 = v/x1
        c2 = v/x2
        y1 = Ostriker99(v,c1,t,rmin,tmin)[0]
        y2 = Ostriker99(v,c2,t,rmin,tmin)[0]
        return [y1+(M-x1)*(y2-y1)/(x2-x1),0]


# def pyvista_volume_density(alpha, x, y, z, cmap='jet', N=10):
#     """
#     Create 3D isosurfaces of log10(alpha) with varying opacity.
    
#     - Masks regions where alpha <= 1e-12
#     - Shows N=10 isosurfaces with values from -1 to +1
#     - Opacity is proportional to 2**value
#     """
#     values = np.log10(alpha).astype(np.float32)
#     Nx, Ny, Nz = values.shape

#     # Create structured grid
#     grid = pv.ImageData()
#     grid.dimensions = (Nx, Ny, Nz)
#     grid.origin = (float(x[0]), float(y[0]), float(z[0]))
#     grid.spacing = (float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0]))
#     grid.point_data["values"] = values.ravel(order="F")
#     grid.point_data["alpha_raw"] = alpha.ravel(order="F")
    
#     # Mask: only keep cells where alpha > 1e-12
#     grid_masked = grid.threshold(value=1e-12, scalars="alpha_raw", method="upper")
    
#     print(f"Original grid: {grid.n_cells:,d} voxels")
#     print(f"After masking (alpha > 1e-12): {grid_masked.n_cells:,d} voxels\n")

#     # Define N=10 isosurface values from -1 to +1
#     N                 = 100
#     isosurface_values = np.linspace(-1, 1.5, N)
    
#     # Colormap range
#     vmin, vmax = -1, 0.75
    
#     # Create plotter
#     p = pv.Plotter()
#     p.add_axes()
#     p.add_box_axes()
    
#     print(f"Creating {N} isosurfaces:\n")
    
#     # Add each isosurface
#     for i, iso_val in enumerate(isosurface_values):
#         # Extract isosurface at this value
#         mesh = grid_masked.contour([iso_val])
        
#         if mesh.n_cells > 0:  # Only add if surface exists
#             # Calculate opacity: proportional to 2**value
#             # Normalize to [0, 1]: opacity = (2**value) / (2 * 2**vmax)
#             opacity = 3**iso_val / (2 * 3**vmax)
#             opacity = np.clip(opacity, 0.0, 1.0)
            
#             # Assign the isosurface value to all points for coloring
#             mesh.point_data["density"] = np.full(mesh.n_points, iso_val)
            
#             p.add_mesh(
#                 mesh,
#                 scalars="density",
#                 cmap=cmap,
#                 clim=(vmin, vmax),
#                 opacity=opacity,
#                 show_scalar_bar=(i == len(isosurface_values) - 1),
#                 scalar_bar_args=dict(title="log10(Alpha)") if i == len(isosurface_values) - 1 else None,
#             )
            
#             print(f"  Isosurface {i+1:2d}/{N}: value={iso_val:7.3f}, 3**value={3**iso_val:6.3f}, opacity={opacity:.3f}")
    
#     print(f"\nDisplaying {N} isosurfaces of log10(Alpha) with opacity ∝ 3**value\n")
#     p.show()




def pyvista_volume_density(alpha, x, y, z, cmap='jet', N=10):
    """
    Create 3D isosurfaces of log10(alpha) with varying opacity.
    
    - Masks regions where alpha <= 1e-12
    - Shows N isosurfaces with values from -1 to +1
    - Opacity is proportional to 3**value
    - Optimized for smooth, fast rendering
    """
    values = np.log10(alpha).astype(np.float32)
    
    # Apply Gaussian smoothing to the field for smoother contours
    values = gaussian_filter(values, sigma=0.0)
    
    Nx, Ny, Nz = values.shape

    # Create structured grid
    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, Nz)
    grid.origin = (float(x[0]), float(y[0]), float(z[0]))
    grid.spacing = (float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0]))
    grid.point_data["values"] = values.ravel(order="F")
    grid.point_data["alpha_raw"] = alpha.ravel(order="F")
    
    # Mask: only keep cells where alpha > 1e-12
    grid_masked = grid.threshold(value=1e-12, scalars="alpha_raw", method="upper")
    
    print(f"Original grid: {grid.n_cells:,d} voxels")
    print(f"After masking (alpha > 1e-12): {grid_masked.n_cells:,d} voxels")
    print(f"Field smoothing: Gaussian sigma=1.5\n")

    # Define N isosurface values from -1 to +1
    isosurface_values = np.linspace(-1, 1.5, N)
    
    # Colormap range
    vmin, vmax = -1, 0.6
    
    # Create plotter with optimizations
    p = pv.Plotter(window_size=(1200, 900))
    p.add_axes()
    p.add_box_axes()
    
    print(f"Creating {N} isosurfaces:\n")
    
    # Add each isosurface
    for i, iso_val in enumerate(isosurface_values):
        # Extract isosurface at this value
        mesh = grid_masked.contour([iso_val])
        
        if mesh.n_cells > 0:  # Only add if surface exists
            # Simplify mesh slightly to reduce polygon count (optional, for speed)
            # Uncomment if rendering is too slow:
            # mesh = mesh.decimate(0.3)  # Keep 30% of triangles
            
            # Calculate opacity: proportional to 3**value
            opacity = 3**iso_val / (2 * 3**vmax)
            opacity = np.clip(opacity, 0.0, 1.0)
            
            # Assign the isosurface value to all points for coloring
            mesh.point_data["density"] = np.full(mesh.n_points, iso_val)
            
            p.add_mesh(
                mesh,
                scalars="density",
                cmap=cmap,
                clim=(vmin, vmax),
                opacity=opacity,
                smooth_shading=True,  # Smoother appearance
                show_scalar_bar=(i == len(isosurface_values) - 1),
                scalar_bar_args=dict(title="log10(Alpha)") if i == len(isosurface_values) - 1 else None,
            )
            
            print(f"  Isosurface {i+1:2d}/{N}: value={iso_val:7.3f}, opacity={opacity:.3f}")
    
    print(f"\nDisplaying {N} isosurfaces (smooth, interactive).\n")
    print("Controls: Scroll/Right-drag=Zoom | Left-drag=Rotate | Middle-drag=Pan | Home=Reset\n")
    p.show()




if __name__ == "__main__":

    load("build/checkpoints/checkpoint_000000.h5")


    #values = np.log10(Alpha + 1e-2)

    pyvista_volume_density(Alpha, Domain.x, Domain.y, Domain.z)



    # ========== 2D Density Map ==========
    plt.contourf(Domain.x, Domain.y, np.log10(Alpha[:,:,Domain.Nz//2] + 1e-1), 1000, vmin=-1, vmax=1, cmap='jet')
    plt.savefig("/Users/davidoneill/Desktop/Echo/Outputs/DensityMap.png", dpi = 400)


    plt.figure()
    plt.plot(Trajectory.Force_t, -2 * Trajectory.Force[:,0])
    plt.plot(Trajectory.Force_t, [Ostriker99(v=2.0, c=1.0, t=t, rmin=0.1, tmin=0.0, GM=1, rho0 = 1)[0] for t in Trajectory.Force_t], c = 'black', linestyle = 'dashed')
    #plt.plot(Trajectory.Force_t, [2 * Trajectory.Force[i,0] / Ostriker99(v=2.0, c=1.0, t=Trajectory.Force_t[i], rmin=0.1, tmin=0.0, GM=1, rho0 = 1) for i in range(0,len(Trajectory.Force_t))])
    plt.savefig("/Users/davidoneill/Desktop/Echo/Outputs/Forces.png", dpi = 400)
