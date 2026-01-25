#include <array>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>


#include "Simulate.hpp"
#include "utils.hpp"


int main() {
    SimConfig cfg{};

    // domain
    cfg.Nx            = 1000; 
    cfg.Ny            = 1000; 
    cfg.Nz            = 1;
    cfg.range_x       = {-3.0, 3.0};
    cfg.range_y       = {-3.0, 3.0};
    cfg.range_z       = {0.0, 0.5};
    cfg.seed_fraction = 0.1;

    // solver
    cfg.cs          = 2.0;
    cfg.rho0        = 1.0;
    cfg.rmin        = 0.05;
    cfg.error_tol   = 1e-8;
    cfg.unique_tol  = 8e-3;
    cfg.max_roots   = 4;

    // runtime
    cfg.timelimiter             = 0.1;
    cfg.finite_timestep         = true;
    cfg.RecordTrajectoryCadence = 0.0001;
    std::size_t Ntraj = 100000;

    // output
    cfg.checkpoint_dir   = "checkpoints";
    cfg.checkpoint_every = 1;

    Simulate sim(cfg);

    // compute times
    std::vector<double> ComputeTimes = linspace(4.0, 5.0, 2);
    auto T = linspace(0.0, 6.0, Ntraj);

    std::vector<double> M(Ntraj, 1.0);
    std::vector<std::array<double,3>> X, V, A;
    X.reserve(Ntraj);
    V.reserve(Ntraj);
    A.reserve(Ntraj);

    double TwoPi = 2.0*3.1415926;

    for (double t : T) {
        M.push_back(1.0);
        X.push_back({std::sin(TwoPi*t), std::cos(TwoPi*t), 0.0});     // x = t
        V.push_back({TwoPi*std::cos(TwoPi*t), -TwoPi*std::sin(TwoPi*t), 0.0});   // constant velocity
        A.push_back({-TwoPi*TwoPi*std::sin(TwoPi*t), -TwoPi*TwoPi*std::cos(TwoPi*t), 0.0});   // zero acceleration
    }

    cfg.time_i = T[0];
    cfg.mass_i = M[0];
    cfg.X_i    = X[0];
    cfg.V_i    = V[0];
    cfg.A_i    = A[0];

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    sim.run_fixed(ComputeTimes, T, M, X, V, A);
    auto t1 = clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    double seconds = dt.count();

    std::cout << "[main] All times have been completed at time "<< seconds << "s\n";
    return 0;
}
