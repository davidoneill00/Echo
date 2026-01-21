#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Domain.hpp"
#include "GaseousDynamics.hpp"
#include "Trajectory.hpp"


struct SimConfig {
  // initial state
  double time_i, mass_i;
  std::array<double,3> X_i, V_i, A_i;

  // domain
  std::size_t Nx, Ny, Nz;
  std::array<double,2> range_x, range_y, range_z;
  double seed_fraction;

  // solver
  double cs, rho0, rmin, error_tol, unique_tol;
  int max_roots;

  // runtime params
  double timelimiter;
  bool finite_timestep;
  int RecordTimeseriesCadence;

  // output
  std::string checkpoint_dir;
  int checkpoint_every;
};


class Simulate {
public:
    Simulate(const SimConfig& sim);

    void run_fixed(
        const std::vector<double>& ComputeTimes,
        const std::vector<double>& TrajectoryTimeseries,
        const std::vector<double>& TrajectoryMasses,
        const std::vector<std::array<double, 3>>& TrajectoryPositions,
        const std::vector<std::array<double, 3>>& TrajectoryVelocities,
        const std::vector<std::array<double, 3>>& TrajectoryAccelerations
    );
private:
  SimConfig cfg;

  LiveTrajectory Trajectory;
  SpatialDomain Domain;
  LinearGasSolver Solver;
};