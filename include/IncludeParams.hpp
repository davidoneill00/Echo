#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>


struct SimConfig {
    int    nx, ny, nz, num_levels, ref_ratio;
    double x_min, x_max, y_min, y_max, z_min, z_max, seed_fraction;
    double cs, rmin, error_tol, unique_tol, max_roots;
    int Ntraj, RecordCadence;
    double time_i, mass_i;
    std::string Xx, Xy, Xz, Vx, Vy, Vz, Ax, Ay, Az; // exprtk expression strings
    double rho0, timelimiter;
    bool live, finite_dt;
    std::string checkpoint_dir;
    int checkpoint_every;  
    std::vector<double> compute_times;
};

inline SimConfig read_input(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open input file: " + path);

    SimConfig p{};
    std::string line, key;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ss >> key;
        if      (key == "nx")               ss >> p.nx;
        else if (key == "ny")               ss >> p.ny;
        else if (key == "nz")               ss >> p.nz;
        else if (key == "seed_fraction")    ss >> p.seed_fraction;
        else if (key == "num_levels")       ss >> p.num_levels;
        else if (key == "ref_ratio")        ss >> p.ref_ratio;
        else if (key == "x_min")            ss >> p.x_min;
        else if (key == "x_max")            ss >> p.x_max;
        else if (key == "y_min")            ss >> p.y_min;
        else if (key == "y_max")            ss >> p.y_max;
        else if (key == "z_min")            ss >> p.z_min;
        else if (key == "z_max")            ss >> p.z_max;
        else if (key == "seed_fraction")    ss >> p.seed_fraction;
        else if (key == "cs")               ss >> p.cs;
        else if (key == "rmin")             ss >> p.rmin;
        else if (key == "error_tol")        ss >> p.error_tol;
        else if (key == "unique_tol")       ss >> p.unique_tol;
        else if (key == "max_roots")        ss >> p.max_roots;
        else if (key == "time_i")           ss >> p.time_i;
        else if (key == "mass_i")           ss >> p.mass_i;
        else if (key == "Xx")               std::getline(ss >> std::ws, p.Xx);
        else if (key == "Xy")               std::getline(ss >> std::ws, p.Xy);
        else if (key == "Xz")               std::getline(ss >> std::ws, p.Xz);
        else if (key == "Vx")               std::getline(ss >> std::ws, p.Vx);
        else if (key == "Vy")               std::getline(ss >> std::ws, p.Vy);
        else if (key == "Vz")               std::getline(ss >> std::ws, p.Vz);
        else if (key == "Ax")               std::getline(ss >> std::ws, p.Ax);
        else if (key == "Ay")               std::getline(ss >> std::ws, p.Ay);
        else if (key == "Az")               std::getline(ss >> std::ws, p.Az);
        else if (key == "Ntraj")            ss >> p.Ntraj;
        else if (key == "RecordCadence")    ss >> p.RecordCadence;
        else if (key == "rho0")             ss >> p.rho0;
        else if (key == "timelimiter")      ss >> p.timelimiter;
        else if (key == "live")             ss >> std::boolalpha >> p.live;
        else if (key == "finite_dt")        ss >> std::boolalpha >> p.finite_dt;
        else if (key == "checkpoint_dir")   ss >> p.checkpoint_dir;
        else if (key == "checkpoint_every") ss >> p.checkpoint_every;
        else if (key == "t") {
            double val;
            while (ss >> val) p.compute_times.push_back(val);
        }
    }
    return p;
}