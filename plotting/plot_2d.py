from read_h5 import load
import matplotlib.pyplot as plt
import numpy as np
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str  , help="Path to the checkpoint file")
    parser.add_argument("--vmin"    , type=float, help="Minimum value for the field", default=-3)
    parser.add_argument("--vmax"    , type=float, help="Maximum value for the field", default= 0)
    parser.add_argument("--cmap"    , type=str  , help="Colormap for the plot"      , default="jet")
    args = parser.parse_args()

    Domain, Trajectory, Params, Alpha, Alpha_t, Levels = load(args.checkpoint)

    # ========== 2D Density Map ==========
    plt.contourf(Domain.x, Domain.y, np.log10(Alpha[:,:,Domain.Nz//2] + 1e-1), 1000, vmin=-1, vmax=1, cmap='jet')
    plt.savefig("/Users/davidoneill/Desktop/Echo/Outputs/DensityMap.png", dpi = 400)






(1) refinement implemented!
(2) plotting options extended
(3) Input file passed at runtime
(4) analytical trajectory supported
(5) Distinction between live/fixed trajectories
(6) finer root inherits coarser roots but will retry when zero