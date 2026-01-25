#include <iostream>

#include "Domain.hpp"
#include "GaseousDynamics.hpp"
#include "RootFinder.hpp"
#include "Trajectory.hpp"
#include "utils.hpp"
#include "CheckpointWriter.hpp"
#include "Simulate.hpp"


Simulate::Simulate(const SimConfig& cfg_in)
  : cfg(cfg_in),
    Trajectory(cfg.time_i, cfg.mass_i, cfg.X_i, cfg.V_i, cfg.A_i, cfg.timelimiter, cfg.finite_timestep, cfg.RecordTrajectoryCadence),
    Domain(cfg.Nx, cfg.Ny, cfg.Nz, cfg.range_x, cfg.range_y, cfg.range_z, cfg.seed_fraction),
    Solver(Trajectory, Domain, cfg.cs, cfg.max_roots, cfg.unique_tol, cfg.error_tol, cfg.rho0, cfg.rmin)
{}

void Simulate::run_fixed(
    const std::vector<double>& ComputeTimes,
    const std::vector<double>& TrajectoryTimeseries,
    const std::vector<double>& TrajectoryMasses,
    const std::vector<std::array<double, 3>>& TrajectoryPositions,
    const std::vector<std::array<double, 3>>& TrajectoryVelocities,
    const std::vector<std::array<double, 3>>& TrajectoryAccelerations){
        
    std::size_t next_idx = 0;
    int checkpoint_index = 0;
    
    for (double t : ComputeTimes) {
        while ((next_idx < TrajectoryTimeseries.size()) && (TrajectoryTimeseries[next_idx] <= t)) {
            
            const double time = TrajectoryTimeseries[next_idx];
            if (time > Trajectory.current().t) {
                Trajectory.add_event(time,
                                        TrajectoryMasses[next_idx],
                                        TrajectoryPositions[next_idx],
                                        TrajectoryVelocities[next_idx],
                                        TrajectoryAccelerations[next_idx]);
            }
            ++next_idx;
        }
    Solver.compute_wake(t);

    
    checkpoint::FixedParams fixed;
    checkpoint::RuntimeParams runtime;
    checkpoint::AlphaSnapshot alpha;

    // fixed parameters for simulation
    fixed.seed_fraction       = cfg.seed_fraction;
    fixed.max_number_of_roots = cfg.max_roots;
    fixed.initial_state       = Trajectory.initial();
    fixed.Nx                  = Domain.Resolution_x; fixed.Ny = Domain.Resolution_y; fixed.Nz = Domain.Resolution_z;
    fixed.range_x             = Domain.extent_x; fixed.range_y = Domain.extent_y; fixed.range_z = Domain.extent_z;
    fixed.sound_speed         = cfg.cs;
    fixed.rho0                = cfg.rho0;
    fixed.rmin                = cfg.rmin;
    fixed.error_tol           = cfg.error_tol;
    fixed.unique_tol          = cfg.unique_tol;

    // runtime options for simulation
    runtime.timelimiter             = cfg.timelimiter;
    runtime.finite_timestep         = cfg.finite_timestep;
    runtime.RecordTrajectoryCadence = cfg.RecordTrajectoryCadence;

    // solution storage
    alpha.Nx      = Domain.Resolution_x;
    alpha.Ny      = Domain.Resolution_y;
    alpha.Nz      = Domain.Resolution_z;
    alpha.t_alpha = t;

    // Alpha is Mat3D with flat storage Alpha.a of size Nx*Ny*Nz
    alpha.alpha_flat = flatten_alpha(Solver.alpha());

    if (checkpoint_index % cfg.checkpoint_every == 0){
        checkpoint::save_checkpoint(
            cfg.checkpoint_dir,
            checkpoint_index,
            fixed,
            runtime,
            Trajectory.times(), Trajectory.masses(),
            Trajectory.positions(), Trajectory.velocities(), Trajectory.accelerations(),
            Trajectory.iteration_count(),
            Trajectory.RecordTrajectory_dt(),
            Solver.force_times(), 
            Solver.force_series(),
            &alpha);

            // const std::string& out_dir,
            // int checkpoint_index,
            // const FixedParams& fixed,
            // const RuntimeParams& runtime,
            // const std::vector<double>& t_series,
            // const std::vector<double>& M_series,
            // const std::vector<std::array<double,3>>& X_series,
            // const std::vector<std::array<double,3>>& V_series,
            // const std::vector<std::array<double,3>>& A_series,
            // int iteration_saved,
            // int RecordTimeseriesCadence_used,
            // const std::vector<double>& force_t,
            // const std::vector<std::array<double,3>>& force_F,
            // const AlphaSnapshot* alpha_snapshot


        std::cout << "[simulate] saved checkpoint number " << checkpoint_index << "\n";
        }
    checkpoint_index++;
    }
}


// 1. Interpolating at start
// 2. Seeding at start
