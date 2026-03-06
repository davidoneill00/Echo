#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

struct OrbitalState {
    double t{};
    double M{};
    std::array<double,3> X{};
    std::array<double,3> V{};
    std::array<double,3> A{};
};

// LiveTrajectory: Built incrementally during ODE integration.
// Use this for trajectories which evolve from a set initial conditions
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

    // for checkpoints
    void rebuild_timeseries(const std::vector<double>& TrajectoryTimeseries,
                            const std::vector<double>& TrajectoryMasses,
                            const std::vector<std::array<double, 3>>& TrajectoryPositions,
                            const std::vector<std::array<double, 3>>& TrajectoryVelocities,
                            const std::vector<std::array<double, 3>>& TrajectoryAccelerations,
                            const int saved_iteration);

    // Most expensive operation in code!
    OrbitalState interpolate(double t) const;

    const OrbitalState&                      initial() const noexcept { return Initial; }
    const OrbitalState&                      current() const noexcept { return Current; }
    const std::vector<double>&               times() const noexcept { return Timeseries; }
    const std::vector<double>&               masses() const noexcept { return Mass; }
    const std::vector<std::array<double,3>>& positions() const noexcept { return Positions; }
    const std::vector<std::array<double,3>>& velocities() const noexcept { return Velocities; }
    const std::vector<std::array<double,3>>& accelerations() const noexcept { return Accelerations; }
    std::size_t                              sample_count() const noexcept { return Timeseries.size(); }
    double                                   mean_dt() const { return (Timeseries.size() >= 2) ? (Timeseries.back()-Timeseries.front()) / double(Timeseries.size()-1) : 0.0; }

    double Timestep_dt(const std::array<double, 3>& Force) const;
    int iteration_count() const noexcept { return iteration; }
    int RecordTrajectoryCadence() const noexcept { return record_cadence_; } // optional

    
private:
    
    // running series
    std::vector<double> Timeseries;
    std::vector<double> Mass;
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::array<double, 3>> Velocities;
    std::vector<std::array<double, 3>> Accelerations;
    
    // Precomputed time step denominators for fast interpolation (one-time cost)
    mutable std::vector<double> denominators_;

    int iteration;
    int record_cadence_;
    double limiter;
    bool require_finite_dt;

    OrbitalState Initial{};
    OrbitalState Current{};

    // Cached index for binary search optimization (mutable for use in const interpolate)
    mutable std::size_t cached_index_ = 0;

    // initial state
    double time_i;
    double mass_i;
    std::array<double, 3> position_i;
    std::array<double, 3> velocity_i;
    std::array<double, 3> acceleration_i;
};


// FixedTrajectory: the full orbit (T, M, X, V, A) is known at construction.
// Use this for analytically prescribed or pre-computed trajectories.
class FixedTrajectory {
public:
    FixedTrajectory(const std::vector<double>&              T,
                    const std::vector<double>&              M,
                    const std::vector<std::array<double,3>>& X,
                    const std::vector<std::array<double,3>>& V,
                    const std::vector<std::array<double,3>>& A);

    OrbitalState interpolate(double t) const;

    const OrbitalState&                       initial()       const noexcept;
    const OrbitalState&                       current()       const noexcept;
    const std::vector<double>&                times()         const noexcept;
    const std::vector<double>&                masses()        const noexcept;
    const std::vector<std::array<double,3>>&  positions()     const noexcept;
    const std::vector<std::array<double,3>>&  velocities()    const noexcept;
    const std::vector<std::array<double,3>>&  accelerations() const noexcept;
    std::size_t                               sample_count()  const noexcept;
    double                                    mean_dt()       const;

    // Exposes the underlying LiveTrajectory for LinearGasSolver,
    // which requires const LiveTrajectory& at its interface.
    const LiveTrajectory& as_live() const noexcept { return traj_; }

private:
    LiveTrajectory traj_;
};
