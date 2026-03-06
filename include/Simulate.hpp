#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Domain.hpp"
#include "GaseousDynamics.hpp"
#include "Trajectory.hpp"
#include "IncludeParams.hpp"


class Simulate {
public:
    Simulate(const SimConfig& cfg);

    // All of T, M, X, V, A are known upfront (pre-computed or analytical).
    // A FixedTrajectory is built internally; no add_event thinning occurs.
    void run_fixed(
        const std::vector<double>&               ComputeTimes,
        const std::vector<double>&               T,
        const std::vector<double>&               M,
        const std::vector<std::array<double,3>>& X,
        const std::vector<std::array<double,3>>& V,
        const std::vector<std::array<double,3>>& A
    );

    //void run_live(
    //);

private:
    SimConfig     cfg;
    RefinedDomain Refined;
};