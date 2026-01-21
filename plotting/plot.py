"""
Echo checkpoint loader

Usage:
    import echo_checkpoint as chk
    chk.load("checkpoints/checkpoint_000123.h5")

    print(chk.Domain.X.shape)
    print(chk.Trajectory.t)
    print(chk.Alpha.data)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Containers (simple namespaces)
# -----------------------------
class _Domain: pass
class _Trajectory: pass
#class _Alpha: pass
class _Params: pass

Domain     = _Domain()
Trajectory = _Trajectory()
#Alpha      = _Alpha()
Params     = _Params()

# -----------------------------
# Loader
# -----------------------------
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
        Trajectory.cadence = f["/trajectory/RecordTimeseriesCadence_used"][()]

        # -------------------------
        # Alpha (optional)
        # -------------------------
        if "/alpha/data" in f:
            Alpha    = f["/alpha/data"][:]   # (Nx,Ny,Nz)
            Alpha_t = f["/alpha/meta/t"][()]
        else:
            Alpha   = None
            Alpha_t = None



load("build/checkpoints/checkpoint_000002.h5")

plt.contourf(Domain.x, Domain.y, np.log10(Alpha[:,:,Domain.Nz//2] + 1e-12), 100, vmin=-1, vmax=1, cmap='jet')
plt.show()