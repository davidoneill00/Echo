
import numpy as np
from read_h5 import load
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
import argparse




def pyvista_amr_density(levels, cmap, vmin, vmax, N=10):
    """
    Composite AMR volume render.

    For each level (coarsest first) the spatial region already covered by any
    finer level is zeroed out, so every voxel is drawn exactly once:
      - finest level  → native fine-grid voxels (small, high-res)
      - coarser levels → native coarse-grid voxels (larger blocks) outside the
                         fine domain (each coarse voxel = ref_ratio^3 fine voxels)

    All levels share the same colormap / isosurface values for a seamless look.
    """
    isosurface_values = np.linspace(vmin, vmax, N)

    p = pv.Plotter(window_size=(1200, 900))
    p.add_axes()
    p.add_box_axes()

    scalar_bar_done = False

    for lev_idx, level in enumerate(levels):
        alpha = np.array(level["data"], dtype=np.float64)   # always a copy
        x, y, z = level["x"], level["y"], level["z"]
        Nx, Ny, Nz = level["Nx"], level["Ny"], level["Nz"]

        # Zero out the region covered by every finer level so it is not drawn twice
        for finer in levels[lev_idx + 1:]:
            xi = np.where((x >= finer["range_x"][0]) & (x <= finer["range_x"][1]))[0]
            yj = np.where((y >= finer["range_y"][0]) & (y <= finer["range_y"][1]))[0]
            zk = np.where((z >= finer["range_z"][0]) & (z <= finer["range_z"][1]))[0]
            if xi.size and yj.size and zk.size:
                alpha[np.ix_(xi, yj, zk)] = 0.0

        values = np.log10(alpha + 1e-30).astype(np.float32)

        grid = pv.ImageData()
        grid.dimensions = (Nx, Ny, Nz)
        grid.origin  = (float(x[0]), float(y[0]), float(z[0]))
        grid.spacing = (float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0]))
        grid.point_data["values"]    = values.ravel(order="F")
        grid.point_data["alpha_raw"] = alpha.ravel(order="F")

        grid_masked = grid.threshold(value=1e-12, scalars="alpha_raw", method="upper")
        print(f"Level {lev_idx}: {grid_masked.n_cells:,d} voxels  "
              f"spacing=({x[1]-x[0]:.4f}, {y[1]-y[0]:.4f}, {z[1]-z[0]:.4f})")

        for i, iso_val in enumerate(isosurface_values):
            mesh = grid_masked.contour([iso_val])
            if mesh.n_cells == 0:
                continue

            opacity = 3**iso_val / (2 * 3**vmax)
            opacity = np.clip(opacity, 0.0, 1.0)
            mesh.point_data["density"] = np.full(mesh.n_points, iso_val)

            show_bar = (not scalar_bar_done) and (lev_idx == len(levels) - 1) \
                       and (i == len(isosurface_values) - 1)
            p.add_mesh(
                mesh,
                scalars="density",
                cmap=cmap,
                clim=(vmin, vmax),
                opacity=opacity,
                smooth_shading=True,
                show_scalar_bar=show_bar,
                scalar_bar_args=dict(title="log10(Alpha)") if show_bar else None,
            )
            if show_bar:
                scalar_bar_done = True

    print("\nAMR composite: finest level full-res, coarser levels fill the outer domain.")
    print("Controls: Scroll/Right-drag=Zoom | Left-drag=Rotate | Middle-drag=Pan | Home=Reset\n")
    p.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str  , help="Path to the checkpoint file")
    parser.add_argument("--vmin"    , type=float, help="Minimum value for the field", default=-1)
    parser.add_argument("--vmax"    , type=float, help="Maximum value for the field", default=0.5)
    parser.add_argument("--cmap"    , type=str  , help="Colormap for the plot"      , default="jet")
    args = parser.parse_args()

    Domain, Trajectory, Params, Alpha, Alpha_t, Levels = load(args.checkpoint)

    pyvista_amr_density(Levels, args.cmap, args.vmin, args.vmax)



