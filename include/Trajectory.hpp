#pragma once

#include <array>
#include <cstddef>
#include <vector>

struct OrbitalState {
    double t{};
    double M{};
    std::array<double,3> X{};
    std::array<double,3> V{};
    std::array<double,3> A{};
};

class LiveTrajectory {
public:
    LiveTrajectory(double time,
                   double mass,
                   const std::array<double, 3>& position,
                   const std::array<double, 3>& velocity,
                   const std::array<double, 3>& acceleration,
                   double timelimiter,
                   bool finite_timestep,
                   double RecordTrajectory_dt);

    void add_event(double time,
                   double mass,
                   const std::array<double, 3>& position,
                   const std::array<double, 3>& velocity,
                   const std::array<double, 3>& acceleration);

    void rebuild_timeseries(const std::vector<double>& TrajectoryTimeseries,
                             const std::vector<double>& TrajectoryMasses,
                             const std::vector<std::array<double, 3>>& TrajectoryPositions,
                             const std::vector<std::array<double, 3>>& TrajectoryVelocities,
                             const std::vector<std::array<double, 3>>& TrajectoryAccelerations,
                             const int saved_iteration);

    OrbitalState interpolate(double t) const;

    void interpolate_into(double t,
                          double& M_out,
                          std::array<double,3>& X_out,
                          std::array<double,3>& V_out,
                          std::array<double,3>& A_out) const
    {
        const std::size_t N = Timeseries.size();

        if (t <= Timeseries.front()) {
            M_out = Mass.front();
            X_out = Positions.front();
            V_out = Velocities.front();
            A_out = Accelerations.front();
            return;
        }
        if (t >= Timeseries.back()) {
            M_out = Mass.back();
            X_out = Positions.back();
            V_out = Velocities.back();
            A_out = Accelerations.back();
            return;
        }

        const double time_reg = Timeseries[N - 2];
        std::size_t i;

        if (t > time_reg) {
            i = N - 2;
        } else {
            // Start search from cached index if plausible
            std::size_t ii = cached_index;
            if (ii >= N - 1) ii = 0;
            
            // Quick check: is t in [Timeseries[ii], Timeseries[ii+1]]?
            if (t >= Timeseries[ii] && t <= Timeseries[ii + 1]) {
                i = ii;
            } else {
                // Fallback: linear search with cached inverse
                const double x = (t - Initial.t) * inv_RecordTrajectoryCadence;
                int ii_int = (int)x;
                if ((double)ii_int > x) --ii_int;
                if (ii_int < 0) ii_int = 0;
                const int max_i = (int)N - 2;
                if (ii_int > max_i) ii_int = max_i;
                i = (std::size_t)ii_int;
            }
            
            cached_index = i;
        }

        const std::size_t j = i + 1;
        const double t0 = Timeseries[i];
        const double t1 = Timeseries[j];
        const double denom = (t1 - t0);
        const double u = (denom != 0.0) ? (t - t0) / denom : 0.0;

        const double M0 = Mass[i];
        const double M1 = Mass[j];
        M_out = M0 + u * (M1 - M0);

        const auto& X0 = Positions[i];
        const auto& X1 = Positions[j];
        const auto& V0 = Velocities[i];
        const auto& V1 = Velocities[j];
        const auto& A0 = Accelerations[i];
        const auto& A1 = Accelerations[j];

        X_out[0] = X0[0] + u * (X1[0] - X0[0]);
        X_out[1] = X0[1] + u * (X1[1] - X0[1]);
        X_out[2] = X0[2] + u * (X1[2] - X0[2]);
        
        V_out[0] = V0[0] + u * (V1[0] - V0[0]);
        V_out[1] = V0[1] + u * (V1[1] - V0[1]);
        V_out[2] = V0[2] + u * (V1[2] - V0[2]);
        
        A_out[0] = A0[0] + u * (A1[0] - A0[0]);
        A_out[1] = A0[1] + u * (A1[1] - A0[1]);
        A_out[2] = A0[2] + u * (A1[2] - A0[2]);
    }

    double Timestep_dt(const std::array<double, 3>& Force) const; 

    const OrbitalState& initial() const noexcept { return Initial; }
    const OrbitalState& current() const noexcept { return Current; }

    const std::vector<double>& times() const noexcept { return Timeseries; }
    const std::vector<double>& masses() const noexcept { return Mass; }
    const std::vector<std::array<double,3>>& positions() const noexcept { return Positions; }
    const std::vector<std::array<double,3>>& velocities() const noexcept { return Velocities; }
    const std::vector<std::array<double,3>>& accelerations() const noexcept { return Accelerations; }

    double RecordTrajectory_dt() const noexcept { return RecordTrajectoryCadence; }
    int iteration_count() const noexcept { return iteration; }
    std::size_t sample_count() const noexcept { return Timeseries.size(); }
    double mean_dt() const { return (Timeseries.size() >= 2) ? (Timeseries.back()-Timeseries.front()) / double(Timeseries.size()-1) : 0.0; }
    // Change?

private:
    
    // running series
    std::vector<double> Timeseries;
    std::vector<double> Mass;
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::array<double, 3>> Velocities;
    std::vector<std::array<double, 3>> Accelerations;

    int iteration;
    double RecordTrajectoryCadence;
    double inv_RecordTrajectoryCadence = 0.0;
    double limiter;
    bool require_finite_dt;

    OrbitalState Initial{};
    OrbitalState Current{};
    
    // Cached state for interpolation hot path
    mutable std::size_t cached_index = 0;

    // initial state
    double time_i;
    double mass_i;
    std::array<double, 3> position_i;
    std::array<double, 3> velocity_i;
    std::array<double, 3> acceleration_i;
};
