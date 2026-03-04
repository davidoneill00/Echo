#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "Simulate.hpp"
#include "utils.hpp"


int main() {
    SimConfig cfg{};
    auto t0 = std::chrono::steady_clock::now();

    // domain
    cfg.Nx            = 256; 
    cfg.Ny            = 256; 
    cfg.Nz            = 512;
    cfg.range_x       = {-4.0, 4.0};
    cfg.range_y       = {-4.0, 4.0};
    cfg.range_z       = {-12.0, 4.0};
    cfg.seed_fraction = 0.1;

    // solver
    cfg.cs          = 0.4;
    cfg.rho0        = 1.0;
    cfg.rmin        = 0.1;
    cfg.error_tol   = 1e-6;
    cfg.unique_tol  = 1e-5;
    cfg.max_roots   = 6;

    // runtime
    cfg.timelimiter             = 0.1;
    cfg.finite_timestep         = true;
    cfg.RecordTrajectoryCadence = 10;

    // output
    cfg.checkpoint_dir = "checkpoints";
    cfg.checkpoint_every = 1;

    Simulate sim(cfg);

    // compute times
    std::size_t Ntraj = 40000;
    std::vector<double> ComputeTimes = {6.0}; //linspace(0.4, 1.0, 1);
    
    auto T = linspace(0.0, 6.01, Ntraj);

    std::vector<double> M(Ntraj, 1.0);
    std::vector<std::array<double,3>> X, V, A;
    X.reserve(Ntraj);
    V.reserve(Ntraj);
    A.reserve(Ntraj);

    double TwoPi = 2 * 3.1415926;
    double Pi    = 3.1415926;
    double Pi2   = Pi / 2.0;    
    double Pi4   = Pi / 4.0;    

    for (double t : T) {
        X.push_back({std::cos(Pi * t), std::sin(Pi * t), 3.0 - 2*t});     
        V.push_back({-Pi * std::sin(Pi * t), Pi * std::cos(Pi * t), -2.0}); 
        A.push_back({-Pi * Pi * std::cos(Pi * t), -Pi * Pi * std::sin(Pi * t), 0.0}); 
    }

    // initial
    cfg.time_i = 0.0;
    cfg.mass_i = 1.0;
    cfg.X_i    = X[0];
    cfg.V_i    = V[0];
    cfg.A_i    = A[0];

    sim.run_fixed(ComputeTimes, T, M, X, V, A);
    //sim.run_fixed(ComputeTimes, T, X, V, A);

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "run_fixed wall time: " << dt.count() << " s\n";
    return 0;
}
