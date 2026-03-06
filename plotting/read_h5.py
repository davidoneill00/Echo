import h5py
import numpy as np
import math



# Load HDF5 files into python readable format
def load(filename):
    class _Domain:     pass
    class _Trajectory: pass
    class _Params:     pass

    Domain     = _Domain()
    Trajectory = _Trajectory()
    Params     = _Params()

    with h5py.File(filename, "r") as f:

        # -------------------------
        # Fixed params / domain
        # -------------------------
        Params.seed_fraction       = f["/params/fixed/seed_fraction"][()]
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

        # Read all refinement levels; each entry is a dict with:
        #   data (Nx,Ny,Nz), x/y/z coordinates, range_x/y/z, t_alpha, level
        num_levels = int(f["/alpha/num_levels"][0])
        Levels = []
        for lev in range(num_levels):
            g = f[f"/alpha/level_{lev}"]
            rx = g["meta/range_x"][:]
            ry = g["meta/range_y"][:]
            rz = g["meta/range_z"][:]
            nx = int(g["meta/Nx"][0])
            ny = int(g["meta/Ny"][0])
            nz = int(g["meta/Nz"][0])
            Levels.append({
                "level":   lev,
                "data":    g["data"][:],
                "t_alpha": float(g["meta/t"][0]),
                "Nx": nx, "Ny": ny, "Nz": nz,
                "range_x": rx, "range_y": ry, "range_z": rz,
                "x": np.linspace(*rx, nx),
                "y": np.linspace(*ry, ny),
                "z": np.linspace(*rz, nz),
            })

        # Expose finest level as the default Alpha (highest resolution near particle)
        finest  = Levels[-1]
        Alpha   = finest["data"]
        Alpha_t = finest["t_alpha"]

        # Update Domain coordinates to match the chosen level
        Domain.x = finest["x"]
        Domain.y = finest["y"]
        Domain.z = finest["z"]

    return Domain, Trajectory, Params, Alpha, Alpha_t, Levels

    





