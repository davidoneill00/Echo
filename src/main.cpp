#include <array>
#include <vector>
#include <iostream>

#include "Simulate.hpp"
#include "utils.hpp"


int main() {
    SimConfig cfg{};

    // initial
    cfg.time_i = 0.0;
    cfg.mass_i = 1.0;
    cfg.X_i    = {-5.0, 0.0, 0.0};
    cfg.V_i    = {1.0, 0.0, 0.0};
    cfg.A_i    = {0.0, 0.0, 0.0};

    // domain
    cfg.Nx = 256; 
    cfg.Ny = 256; 
    cfg.Nz = 64;
    cfg.range_x       = {-5.0, 5.0};
    cfg.range_y       = {-5.0, 5.0};
    cfg.range_z       = {-3.0, 3.0};
    cfg.seed_fraction = 0.05;

    // solver
    cfg.cs          = 0.5;
    cfg.rho0        = 1.0;
    cfg.rmin        = 0.05;
    cfg.error_tol   = 1e-6;
    cfg.unique_tol  = 1e-5;
    cfg.max_roots   = 2;

    // runtime
    cfg.timelimiter             = 0.1;
    cfg.finite_timestep         = true;
    cfg.RecordTimeseriesCadence = 1000;

    // output
    cfg.checkpoint_dir = "checkpoints";
    cfg.checkpoint_every = 1;

    Simulate sim(cfg);

    // compute times
    std::vector<double> ComputeTimes = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    std::size_t Ntraj = 2000;

    auto T = linspace(0.0, 6.0, Ntraj);

    std::vector<double> M(Ntraj, 1.0);
    std::vector<std::array<double,3>> X, V, A;
    X.reserve(Ntraj);
    V.reserve(Ntraj);
    A.reserve(Ntraj);

    for (double t : T) {
        X.push_back({t, 0.0, 0.0});     // x = t
        V.push_back({1.0, 0.0, 0.0});   // constant velocity
        A.push_back({0.0, 0.0, 0.0});   // zero acceleration
    }

    sim.run_fixed(ComputeTimes, T, M, X, V, A);

    std::cout << "Done.\n";
    return 0;
}
