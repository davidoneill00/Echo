"""
Echo checkpoint loader
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Containers (simple namespaces)
# -----------------------------
class _Domain: pass
class _Trajectory: pass
class _Params: pass

Domain     = _Domain()
Trajectory = _Trajectory()
Params     = _Params()

# -----------------------------
# Loader
# -----------------------------
def load(filename):
    global Domain, Trajectory, Alpha, Params, Force

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
        Trajectory.cadence = f["/trajectory/RecordTrajectoryCadence_used"][()]

        # -------------------------
        # Alpha (optional)
        # -------------------------
        if "/alpha/data" in f:
            Alpha    = f["/alpha/data"][:]   # (Nx,Ny,Nz)
            Alpha_t = f["/alpha/meta/t"][()]
        else:
            Alpha   = None
            Alpha_t = None

        
        Trajectory.F   = f["/force/F"][:]
        Trajectory.F_t = f["/force/t"][:]

def Ostriker99(v, c, t, rmin=0.05):
    Mach = v/c
    I    = 0.5 * np.log10(1 - 1/Mach/Mach) + np.log10(v*t/rmin)
    F    = 4 * np.pi * I / Mach / Mach 
    return F

load("build/checkpoints/checkpoint_000001.h5")

#plt.plot(Trajectory.X[:,0], Trajectory.X[:,1])
# plt.plot(Trajectory.t, Trajectory.X[:,0])
# plt.plot(Trajectory.t, Trajectory.X[:,1])
# plt.show()

plt.contourf(Domain.x, Domain.y, np.log10(Alpha[:,:,0] + 1e-2), levels = np.linspace(-2, 1, 1000), cmap='jet')
plt.colorbar()
plt.show()

# plt.title('Force Series')
# plt.plot(Trajectory.F_t, Ostriker99(1, 0.5, Trajectory.F_t), c = 'black')
# plt.plot(Trajectory.F_t, Trajectory.F)
# plt.show()



#plt.savefig('/Users/davidoneill/Desktop/ForceTimeseries1.pdf', dpi = 400)

# crash mzps: 1024x1024x512
# 828.319s  : 512x512x256
# 115.068s  : 256x256x128
# 17.3158s  : 128x128x64


# Upsample procedure: Split entire domain into N^3 cells per voxel
#                  (1) Far from individual perturber, have uniform split
#                  (2) Near each perturber we call compute on much smaller region and embed
#                  (3) Now have AMR on Nx*Ny*Nz*N^3 easuly able to resolve small and large scales!