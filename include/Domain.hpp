#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "utils.hpp" // linspace()

class SpatialDomain {
public:
    std::size_t Resolution_x;
    std::size_t Resolution_y;
    std::size_t Resolution_z;

    std::array<double, 2> extent_x;
    std::array<double, 2> extent_y;
    std::array<double, 2> extent_z;

    double seed_fraction;
    std::size_t increment;

    std::vector<double> X, Y, Z;
    std::vector<double> Seed_x, Seed_y, Seed_z;
    std::vector<std::array<int,3>> SeedOrigins;

    SpatialDomain(std::size_t Nx,
           std::size_t Ny,
           std::size_t Nz,
           std::array<double, 2> range_x,
           std::array<double, 2> range_y,
           std::array<double, 2> range_z,
           double seeds)
        : Resolution_x(Nx),
          Resolution_y(Ny),
          Resolution_z(Nz),
          extent_x(range_x),
          extent_y(range_y),
          extent_z(range_z),
          seed_fraction(seeds)
    {
        if (seed_fraction <= 0.0) throw std::runtime_error("SeedFraction must be > 0");
        increment = static_cast<std::size_t>(1.0 / seed_fraction);
        if (increment < 1) increment = 1;

        X = linspace(extent_x[0], extent_x[1], Resolution_x);
        Y = linspace(extent_y[0], extent_y[1], Resolution_y);
        Z = linspace(extent_z[0], extent_z[1], Resolution_z);

        std::size_t nx_s = (Resolution_x + increment - 1) / increment;
        std::size_t ny_s = (Resolution_y + increment - 1) / increment;
        std::size_t nz_s = (Resolution_z + increment - 1) / increment;

        SeedOrigins.reserve(nx_s * ny_s * nz_s);
        Seed_x.reserve(nx_s);
        Seed_y.reserve(ny_s);
        Seed_z.reserve(nz_s);

        for (std::size_t k = 0; k < Resolution_z; k += increment) {
            Seed_z.push_back(Z[k]);
            for (std::size_t j = 0; j < Resolution_y; j += increment) {
                if (k == 0) Seed_y.push_back(Y[j]);
                for (std::size_t i = 0; i < Resolution_x; i += increment) {
                    if (k == 0 && j == 0) Seed_x.push_back(X[i]);
                    SeedOrigins.push_back({static_cast<int>(i),
                                           static_cast<int>(j),
                                           static_cast<int>(k)});
                }
            }
        }
    }
};



// How do we have adaptive seed searching?
// -> In NBody integrator, choose initial position of perturber