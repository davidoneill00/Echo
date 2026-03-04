#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <unordered_map>

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
    std::array<double, 3> dX;

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
        this->dX = {
            (Resolution_x > 1) ? (X[1] - X[0]) : 0.0,
            (Resolution_y > 1) ? (Y[1] - Y[0]) : 0.0,
            (Resolution_z > 1) ? (Z[1] - Z[0]) : 0.0
        };
    }
};




class RefinedDomain {
public:
    std::size_t Level_Resolution_x;
    std::size_t Level_Resolution_y;
    std::size_t Level_Resolution_z;

    std::array<double, 2> Global_extent_x;
    std::array<double, 2> Global_extent_y;
    std::array<double, 2> Global_extent_z;

    double seed_fraction;
    int    num_levels;
    int    refinement_ratio;

    std::unordered_map<int, SpatialDomain> LevelMap;

    RefinedDomain(std::size_t nx,
                  std::size_t ny,
                  std::size_t nz,
                  std::array<double, 2> range_x,
                  std::array<double, 2> range_y,
                  std::array<double, 2> range_z,
                  double seeds,
                  int n_levels,
                  int ref_ratio)
        : Level_Resolution_x(nx),
          Level_Resolution_y(ny),
          Level_Resolution_z(nz),
          Global_extent_x(range_x),
          Global_extent_y(range_y),
          Global_extent_z(range_z),
          seed_fraction(seeds),
          num_levels(n_levels),
          refinement_ratio(ref_ratio)
    {
        if (ref_ratio < 2)
            throw std::runtime_error("refinement_ratio must be >= 2.");
        // Fine-level boundaries land on coarse grid nodes only if Level_Resolution
        // is divisible by refinement_ratio in each dimension.
        if (nx % ref_ratio != 0 || ny % ref_ratio != 0 || nz % ref_ratio != 0)
            throw std::runtime_error(
                "Level_Resolution must be divisible by refinement_ratio "
                "so that coarse-fine boundaries align with coarse grid nodes.");
    }

    bool CheckParticleInDomain(const std::array<double, 3>& Particle_Position,
                               const SpatialDomain& Domain) const {
        return (Particle_Position[0] >= Domain.extent_x[0] && Particle_Position[0] <= Domain.extent_x[1]) &&
               (Particle_Position[1] >= Domain.extent_y[0] && Particle_Position[1] <= Domain.extent_y[1]) &&
               (Particle_Position[2] >= Domain.extent_z[0] && Particle_Position[2] <= Domain.extent_z[1]);
    }

    // Returns the continuous (fractional) grid index of the particle.
    // Useful for field interpolation at the particle location.
    std::array<double, 3> LocateParticleInDomain(
        const SpatialDomain& Domain,
        const std::array<double, 3>& Particle_Position) const
    {
        if (!CheckParticleInDomain(Particle_Position, Domain))
            throw std::runtime_error("Particle position is outside of the domain extent.");

        return { (Particle_Position[0] - Domain.X[0]) / Domain.dX[0],
                 (Particle_Position[1] - Domain.Y[0]) / Domain.dX[1],
                 (Particle_Position[2] - Domain.Z[0]) / Domain.dX[2] };
    }

    // Computes the fine-level extent centred on the particle, with boundaries
    // snapped to coarse grid nodes.  This guarantees:
    //   (1) Cell-size ratio between adjacent levels is exactly refinement_ratio.
    //   (2) Coarse–fine interfaces coincide with coarse cell faces (no misaligned
    //       transitions, satisfies the 2:1 balance rule when ref_ratio == 2).
    //   (3) The particle always lies inside the fine domain.
    //
    // The fine domain spans  (N / refinement_ratio)  coarse cells in each
    // dimension, so its physical width shrinks by 1/refinement_ratio each level
    // while keeping the same number of grid points.
    std::array<std::array<double, 2>, 3> ComputeNewExtent(
        const SpatialDomain& CoarseDomain,
        const std::array<double, 3>& Particle_Position) const
    {
        const std::size_t N[3] = { Level_Resolution_x,
                                   Level_Resolution_y,
                                   Level_Resolution_z };
        const std::vector<double>* axes[3] = { &CoarseDomain.X,
                                               &CoarseDomain.Y,
                                               &CoarseDomain.Z };
        std::array<std::array<double, 2>, 3> NewExtent;

        for (int d = 0; d < 3; ++d) {
            double dx_c   = CoarseDomain.dX[d];
            double origin = (*axes[d])[0];

            // Number of coarse cells spanned by the fine domain.
            int span = static_cast<int>(N[d]) / refinement_ratio;

            // Snap to the nearest coarse grid node to the particle, then
            // place the fine domain symmetrically around that node.
            double pos_rel = Particle_Position[d] - origin;
            int i_centre   = static_cast<int>(std::round(pos_rel / dx_c));
            int i_lo       = i_centre - span / 2;

            NewExtent[d][0] = origin + i_lo * dx_c;
            NewExtent[d][1] = origin + (i_lo + span) * dx_c; // == [0] + N*dx_fine
        }
        return NewExtent;
    }

    // Checks that the fine extent is strictly inside the coarse domain with at
    // least one coarse cell of buffer on every face (proper nesting criterion).
    void CheckProperNesting(const SpatialDomain& CoarseDomain,
                            const std::array<std::array<double, 2>, 3>& FineExtent) const
    {
        const std::array<double, 2>* extents[3] = { &CoarseDomain.extent_x,
                                                     &CoarseDomain.extent_y,
                                                     &CoarseDomain.extent_z };
        for (int d = 0; d < 3; ++d) {
            double buf = CoarseDomain.dX[d]; // one coarse cell on each side
            if (FineExtent[d][0] < (*extents[d])[0] + buf ||
                FineExtent[d][1] > (*extents[d])[1] - buf)
                throw std::runtime_error(
                    "Proper nesting violated: particle is too close to the coarse "
                    "domain boundary. Increase the coarse domain extent or reduce "
                    "num_levels.");
        }
    }

    SpatialDomain ConstructDomainAtNextLevel(
        const SpatialDomain& Previous_Domain,
        const std::array<double, 3>& Particle_Position)
    {
        if (!CheckParticleInDomain(Particle_Position, Previous_Domain))
            throw std::runtime_error("Particle position is outside of the domain extent.");

        std::array<std::array<double, 2>, 3> NewExtent =
            ComputeNewExtent(Previous_Domain, Particle_Position);

        CheckProperNesting(Previous_Domain, NewExtent);

        return SpatialDomain(Level_Resolution_x,
                             Level_Resolution_y,
                             Level_Resolution_z,
                             NewExtent[0],
                             NewExtent[1],
                             NewExtent[2],
                             seed_fraction);
    }

    void initialize_levels(const std::array<double, 3>& Particle_Position) {
        SpatialDomain Previous_Domain = SpatialDomain(
            Level_Resolution_x, Level_Resolution_y, Level_Resolution_z,
            Global_extent_x, Global_extent_y, Global_extent_z,
            seed_fraction);
        LevelMap.insert_or_assign(0, Previous_Domain);

        for (int n_level = 1; n_level < num_levels; ++n_level) {
            SpatialDomain Next_Level = ConstructDomainAtNextLevel(Previous_Domain, Particle_Position);
            LevelMap.insert_or_assign(n_level, Next_Level);
            Previous_Domain          = Next_Level;
        }
    }
};

