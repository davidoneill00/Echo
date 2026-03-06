#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Trajectory.hpp"  // for OrbitalState

namespace checkpoint {

struct FixedParams {
    double seed_fraction       = 0.0;
    int    max_number_of_roots = 0;
    OrbitalState initial_state{};

    std::size_t Nx = 0, Ny = 0, Nz = 0;
    std::array<double,2> range_x{}, range_y{}, range_z{};

    double sound_speed = 0.0;
    double rho0        = 0.0;
    double rmin        = 0.0;
    double error_tol   = 0.0;
    double unique_tol  = 0.0;
};

struct RuntimeParams {
    double timelimiter             = 0.0;
    bool   finite_timestep         = true;
    double RecordTrajectoryCadence = 0.001;
};

struct AlphaSnapshot {
    std::vector<double> alpha_flat;
    std::size_t Nx = 0, Ny = 0, Nz = 0;
    double t_alpha = 0.0;
    int level = 0;
    std::array<double,2> range_x{}, range_y{}, range_z{};
};

inline std::string checkpoint_filename(const std::string& dir, int index) {
    std::ostringstream oss;
    oss << dir;
    if (!dir.empty() && dir.back() != '/') oss << "/";
    oss << "checkpoint_" << std::setw(6) << std::setfill('0') << index << ".h5";
    return oss.str();
}

inline std::string checkpoint_tmp_filename(const std::string& dir, int index) {
    std::ostringstream oss;
    oss << dir;
    if (!dir.empty() && dir.back() != '/') oss << "/";
    oss << "checkpoint_" << std::setw(6) << std::setfill('0') << index << ".tmp.h5";
    return oss.str();
}

inline std::vector<double> flatten_vec3(const std::vector<std::array<double,3>>& v) {
    std::vector<double> out(v.size() * 3);
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[3*i + 0] = v[i][0];
        out[3*i + 1] = v[i][1];
        out[3*i + 2] = v[i][2];
    }
    return out;
}

} // namespace checkpoint
