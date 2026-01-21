#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <vector>

#include "utils.hpp"       // Mat4D, Mat3D, Mat2I, idx3d/idx4d, interp_linear_uniform (if used in Alpha3d)
#include "Trajectory.hpp"  // LiveTrajectory, OrbitalState
#include "Domain.hpp"


// ---------------------------------------------------------------
// Core algorithms
// ---------------------------------------------------------------
bool propagate_full(
    double tr,
    double t,
    const std::array<double,3>& X,
    const std::array<double,3>& dX,
    double cs,
    double error_tol,
    const LiveTrajectory& traj,
    double* tau_out,
    double* err_out);

void bfs3d(
    Mat4D& Roots,
    Mat4D& Errors,
    const std::vector<double>& X,
    const std::vector<double>& Y,
    const std::vector<double>& Z,
    const Mat2I& Neighbours,
    double t,
    double sound_speed,
    double error_tol,
    double unique_tol,
    const LiveTrajectory& traj);

void Alpha3d(
    Mat3D& Alpha,
    Mat3D& NRoots,
    const Mat4D& Roots,
    const std::vector<double>& X,
    const std::vector<double>& Y,
    const std::vector<double>& Z,
    const LiveTrajectory& traj,
    const double cs);

// ---------------------------------------------------------------
// Gas force + integration
// ---------------------------------------------------------------
std::array<double,3> ComputeGasForce(
    const OrbitalState& p,
    const SpatialDomain& domain,
    const Mat3D& Alpha,
    double rho_0,
    double rmin);

// void RK2_ParticleIntegrator(
//     std::vector<Particle>& particles,
//     double dt,
//     double t0);

class LinearGasSolver {
public:
    LinearGasSolver(const LiveTrajectory& traj,
           const SpatialDomain& domain,
           double cs,
           int Nroots,
           double unique_tolerance,
           double error_tolerance,
           double rho,
           double rminimum);

    void compute_wake(double t);

    const Mat3D& alpha() const noexcept { return Alpha; }

    const std::array<double,3>& current_force() const noexcept { return CurrentForce; }
    const std::vector<std::array<double,3>>& force_series() const noexcept { return ForceSeries; }


private:
    const LiveTrajectory& Trajectory;
    const SpatialDomain& Domain;
    double sound_speed;
    double rmin;
    double unique_tol;
    double error_tol;
    double rho_0;
    int Max_Number_of_Roots;

    Mat4D Roots, Errors;
    Mat3D Alpha, NRoots;

    void ComputeSeedRoots(double t);
    double RootFunction(double tr, double t, const std::array<double,3> X);
    std::vector<double> RootValues(double t, const std::array<double,3> X, double a, double b);
    std::array<double,3> CurrentForce{};
    std::vector<std::array<double,3>> ForceSeries;
};
