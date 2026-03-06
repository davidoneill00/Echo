#include <iostream>
#include <memory>

#include "Domain.hpp"
#include "GaseousDynamics.hpp"
#include "RootFinder.hpp"
#include "Trajectory.hpp"
#include "utils.hpp"
#include "CheckpointWriter.hpp"
#include "Simulate.hpp"
#include "IncludeParams.hpp"


Simulate::Simulate(const SimConfig& cfg_in)
    : cfg(cfg_in),
      Refined(cfg_in.nx, cfg_in.ny, cfg_in.nz,
             {cfg_in.x_min, cfg_in.x_max},
             {cfg_in.y_min, cfg_in.y_max},
             {cfg_in.z_min, cfg_in.z_max},
              cfg_in.seed_fraction,
              cfg_in.num_levels,
              cfg_in.ref_ratio)
{}

void Simulate::run_fixed(
    const std::vector<double>&               ComputeTimes,
    const std::vector<double>&               T,
    const std::vector<double>&               M,
    const std::vector<std::array<double,3>>& X,
    const std::vector<std::array<double,3>>& V,
    const std::vector<std::array<double,3>>& A)
{
    FixedTrajectory traj(T, M, X, V, A);

    int checkpoint_index = 0;
    std::vector<double>               force_t_out;
    std::vector<std::array<double,3>> force_F_out;

    for (double t : ComputeTimes) {

        // Rebuild refined domain centered on particle's position at this compute time
        OrbitalState state = traj.interpolate(t);
        Refined.initialize_levels(state.X);

        // Solve at each level in order (0 = coarsest, num_levels-1 = finest).
        // Force is taken from the finest level (best near-field resolution).
        std::vector<checkpoint::AlphaSnapshot> alpha_snapshots;


        // prev_solver is kept alive so the next (finer) level can seed from it.
        std::unique_ptr<LinearGasSolver> prev_solver;
        int prev_lev = -1;

        for (int lev = 0; lev < cfg.num_levels; ++lev) {
            auto it = Refined.LevelMap.find(lev);
            if (it == Refined.LevelMap.end()) continue;
            const SpatialDomain& domain = it->second;

            auto Solver = std::make_unique<LinearGasSolver>(
                traj.as_live(), domain,
                cfg.cs, static_cast<int>(cfg.max_roots),
                cfg.unique_tol, cfg.error_tol, cfg.rho0, cfg.rmin);

            if (lev > 0 && prev_solver) {
                // Seed every fine-grid voxel from the nearest coarse-grid voxel,
                // then release the coarse solver before BFS to recover memory.
                const SpatialDomain& coarse_domain = Refined.LevelMap.at(prev_lev);
                Solver->seed_from_coarse(coarse_domain, *prev_solver);
                prev_solver.reset();
                Solver->compute_wake(t, /*use_existing_seeds=*/true);
            } else {
                Solver->compute_wake(t);
            }

            checkpoint::AlphaSnapshot snap;
            snap.Nx         = domain.Resolution_x;
            snap.Ny         = domain.Resolution_y;
            snap.Nz         = domain.Resolution_z;
            snap.t_alpha    = t;
            snap.level      = lev;
            snap.range_x    = domain.extent_x;
            snap.range_y    = domain.extent_y;
            snap.range_z    = domain.extent_z;
            snap.alpha_flat = flatten_alpha(Solver->alpha());
            alpha_snapshots.push_back(std::move(snap));

            // Keep this solver alive to seed the next (finer) level
            prev_lev    = lev;
            prev_solver = std::move(Solver);
        }

        // Composite force: each level contributes only from voxels that are NOT
        // covered by any finer level, so the integration region is non-overlapping.
        // (Simply taking the finest level's force would miss the outer coarse domain.)
        {
            std::array<double,3> composite_force{};
            OrbitalState p_now = traj.interpolate(t);

            // loop over levels
            for (int si = 0; si < (int)alpha_snapshots.size(); ++si) {
                const auto& snap = alpha_snapshots[si];
                const SpatialDomain& dom = Refined.LevelMap.at(snap.level);

                // Restore alpha into a Mat3D (flatten_alpha uses the same row-major layout)
                Mat3D alpha_m((int)snap.Nx, (int)snap.Ny, (int)snap.Nz);
                std::copy(snap.alpha_flat.begin(), snap.alpha_flat.end(), alpha_m.a.begin());

                // Zero voxels covered by any finer level
                for (int fj = si + 1; fj < (int)alpha_snapshots.size(); ++fj) {
                    const auto& finer = alpha_snapshots[fj];
                    for (int i = 0; i < (int)snap.Nx; ++i) {
                        if (dom.X[i] < finer.range_x[0] || dom.X[i] > finer.range_x[1]) continue;
                        for (int j = 0; j < (int)snap.Ny; ++j) {
                            if (dom.Y[j] < finer.range_y[0] || dom.Y[j] > finer.range_y[1]) continue;
                            for (int k = 0; k < (int)snap.Nz; ++k) {
                                if (dom.Z[k] < finer.range_z[0] || dom.Z[k] > finer.range_z[1]) continue;
                                alpha_m(i, j, k) = 0.0;
                            }
                        }
                    }
                }

                auto fvec = ComputeGasForce(p_now, dom, alpha_m, cfg.rho0, cfg.rmin);
                for (int d = 0; d < 3; ++d) composite_force[d] += fvec[d];
            }

            force_t_out.push_back(t);
            force_F_out.push_back(composite_force);
        }

        // Base (level 0) domain used for the fixed params header
        const SpatialDomain& base = Refined.LevelMap.at(0);

        checkpoint::FixedParams fixed;
        fixed.seed_fraction       = cfg.seed_fraction;
        fixed.max_number_of_roots = static_cast<int>(cfg.max_roots);
        fixed.initial_state       = traj.initial();
        fixed.Nx                  = base.Resolution_x;
        fixed.Ny                  = base.Resolution_y;
        fixed.Nz                  = base.Resolution_z;
        fixed.range_x             = base.extent_x;
        fixed.range_y             = base.extent_y;
        fixed.range_z             = base.extent_z;
        fixed.sound_speed         = cfg.cs;
        fixed.rho0                = cfg.rho0;
        fixed.rmin                = cfg.rmin;
        fixed.error_tol           = cfg.error_tol;
        fixed.unique_tol          = cfg.unique_tol;

        checkpoint::RuntimeParams runtime;
        runtime.timelimiter             = cfg.timelimiter;
        runtime.finite_timestep         = cfg.finite_dt;
        runtime.RecordTrajectoryCadence = cfg.RecordCadence;

        if (checkpoint_index % cfg.checkpoint_every == 0) {
            checkpoint::save_checkpoint(
                cfg.checkpoint_dir,
                checkpoint_index,
                fixed,
                runtime,
                traj.times(),
                traj.masses(),
                traj.positions(),
                traj.velocities(),
                traj.accelerations(),
                static_cast<int>(traj.sample_count()),
                cfg.RecordCadence,
                force_t_out,
                force_F_out,
                alpha_snapshots);
            std::cout << "[time " << t << "] Wrote checkpoint to: " << cfg.checkpoint_dir
                      << " (" << alpha_snapshots.size() << " levels)\n";
        }
        checkpoint_index++;
    }
}


//void Simulate::run_live(){};