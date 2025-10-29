from mpi4py import MPI
import numpy as np
from Method import Walk
from bfs_solver import UniqueArray_4d, Alpha3d

comm = MPI.COMM_WORLD      # setup global comm between ranks
rank = comm.Get_rank()     # label each rank 
size = comm.Get_size()     # total number of ranks

def distribute_list(data, rank, size):
    return data[rank::size]

def ComputeSeedRoots(Domain, Method, my_cubes, t):
    local_origin_roots = []
    for origin in my_cubes:
        ix, iy, iz    = origin
        x, y, z       = Domain.X[ix], Domain.Y[iy], Domain.Z[iz]
        a, b          = Method.Trajectory.Time_i, t
        f_a, f_b      = Method.RootFunction([a, b], t, x, y, z)
        roots         = Method.RootValues(t, x, y, z, a, b, f_a, f_b)
        # Note: for multiple particle perturbers we do this for each one and keep track of each trajectory

        local_origin_roots.append((origin,roots))

    return local_origin_roots


def assemble_global_field(results, Domain, EmptyRoots):
    """
    Combine all 3D root arrays into one global 4D array.
    
    Parameters
    ----------
    results     : combined lists of roots from each rank
    Domain      : object describing spatial grid
    EmptyRoots  : empty array of some desired shape


    Returns
    -------
    GlobalRoots : array -> (i,j,k,r) denoting spatial and root indices
    """

    rootnumber = 0
    for Roots in results:
        EmptyRoots[:,:,:,rootnumber]  = Roots[:, :, :]
        rootnumber                   += 1

    GlobalRoots = np.copy(EmptyRoots)
    return GlobalRoots



def save_global_field(GlobalAlpha, Domain, save_path="results_mpi/global_field.h5"):
    """Save the assembled global Alpha field to a HDF5 file."""
    import h5py, os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("Alpha_Global", data=GlobalAlpha)
        f.attrs["shape"] = GlobalAlpha.shape
        f.attrs["Resolution_x"] = Domain.Resolution_x
        f.attrs["Resolution_y"] = Domain.Resolution_y
        f.attrs["Resolution_z"] = Domain.Resolution_z

    print(f"[Rank 0] Saved global Alpha field to {save_path} (shape={GlobalAlpha.shape})", flush=True)


def run_mpi(Domain, Method, t, error_tol):
    """
    Parallel BFS propagation across different roots using MPI.

    Parameters
    ----------
    Domain : spatial grid object
    Method : DensityWakeSolver object
    t      : Evaluation time.

    """
    
    # === 1. Distribute the seeds ===
    origins  = Domain.SeedOrigins
    #if rank == 0:
    #    [print(o) for o in origins]
    
    my_cubes = distribute_list(origins, rank, size)

    # === 2. Compute the seeds on each rank ===
    local_origin_roots = ComputeSeedRoots(Domain, Method, my_cubes, t)

    # === 3. Global communication of all seeds ===
    all_origin_roots      = comm.allgather(local_origin_roots)
    all_origin_roots_flat = [pair for sublist in all_origin_roots for pair in sublist] # flattening

    # === 4. Identify the ``richest seed'' ===
    if rank == 0:
        richest_seed, richest_roots = max(all_origin_roots_flat,key=lambda item: np.isfinite(item[1]).sum())
    else:
        richest_seed, richest_roots = None, None

    richest_seed  = comm.bcast(richest_seed, root=0)
    richest_roots = comm.bcast(richest_roots, root=0)
    my_roots      = distribute_list(richest_roots, rank, size)


    # === 5. Each rank will BFS the root it has been given ===
    
    local_results = []
    for tr in my_roots:  
        walker       = Walk(Domain, Method, t)
        Roots        = walker.Cube(seed_indx=richest_seed, seed_root=tr, error_tol=error_tol)[0]
        local_results.append(Roots)



    # === 6. Gather results on rank 0 ===
    gathered = comm.gather(local_results, root=0)
    
    if rank == 0:
        Nx, Ny, Nz  = Domain.Resolution_x, Domain.Resolution_y, Domain.Resolution_z
        Nroots      = len(richest_roots)
        EmptyAlpha  = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        EmptyRoots  = np.full((Nx, Ny, Nz, Nroots), np.nan, dtype=np.float64)

        all_results = [item for sublist in gathered for item in sublist]
        GlobalRoots = assemble_global_field(all_results, Domain, EmptyRoots)

        # === 7. Pick out unique roots for each (i,j,k) ===
        unique_tol  = 5e-12
        UniqueRoots = UniqueArray_4d(GlobalRoots, unique_tol)
        
        # === 7.5. check other seeds! ===
        # ignore for now

        # === 8. Use unique root grid to compute density perturbations ===
        GlobalAlpha, NRoots = Alpha3d(
            UniqueRoots,
            Domain.X,
            Domain.Y,
            Domain.Z,
            Method._t_arr,
            Method._x_arr,
            Method._v_arr,
            Method.sound_speed
            )

        

        print(f"[MPI] Global Alpha field assembled. Shape = {GlobalAlpha.shape}")
        return GlobalAlpha, NRoots

    return None, None






