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
                   int cadence);

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

    double Timestep_dt(const std::array<double, 3>& Force) const;

    const OrbitalState& initial() const noexcept { return Initial; }
    const OrbitalState& current() const noexcept { return Current; }

    const std::vector<double>& times() const noexcept { return Timeseries; }
    const std::vector<double>& masses() const noexcept { return Mass; }
    const std::vector<std::array<double,3>>& positions() const noexcept { return Positions; }
    const std::vector<std::array<double,3>>& velocities() const noexcept { return Velocities; }
    const std::vector<std::array<double,3>>& accelerations() const noexcept { return Accelerations; }

    int iteration_count() const noexcept { return iteration; }
    int RecordTimeseriesCadence() const noexcept { return record_cadence_; } // optional

    std::size_t sample_count() const noexcept { return Timeseries.size(); }

    double mean_dt() const { return (Timeseries.size() >= 2) ? (Timeseries.back()-Timeseries.front()) / double(Timeseries.size()-1) : 0.0; }

private:
    
    // running series
    std::vector<double> Timeseries;
    std::vector<double> Mass;
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::array<double, 3>> Velocities;
    std::vector<std::array<double, 3>> Accelerations;

    int iteration;
    int record_cadence_;
    double limiter;
    bool require_finite_dt;

    OrbitalState Initial{};
    OrbitalState Current{};

    // initial state
    double time_i;
    double mass_i;
    std::array<double, 3> position_i;
    std::array<double, 3> velocity_i;
    std::array<double, 3> acceleration_i;
};
