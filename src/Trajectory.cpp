#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>

#include "Trajectory.hpp"
#include "utils.hpp"


// ---------------------------------------------------------------
// FixedTrajectory
// ---------------------------------------------------------------

FixedTrajectory::FixedTrajectory(
    const std::vector<double>&               T,
    const std::vector<double>&               M,
    const std::vector<std::array<double,3>>& X,
    const std::vector<std::array<double,3>>& V,
    const std::vector<std::array<double,3>>& A)
    // Construct LiveTrajectory with the first point; rebuild_timeseries
    // then replaces all internal vectors with the complete arrays
    : traj_(T.at(0), M.at(0), X.at(0), V.at(0), A.at(0),
            /*timelimiter=*/0.0, /*finite_timestep=*/false, /*cadence=*/1)
{
    if (T.size() < 2)
        throw std::runtime_error("FixedTrajectory requires at least 2 points.");
    traj_.rebuild_timeseries(T, M, X, V, A, static_cast<int>(T.size()));
}

OrbitalState FixedTrajectory::interpolate(double t) const { return traj_.interpolate(t); }

const OrbitalState& FixedTrajectory::initial() const noexcept { return traj_.initial(); }
const OrbitalState& FixedTrajectory::current() const noexcept { return traj_.current(); }

const std::vector<double>& FixedTrajectory::times()         const noexcept { return traj_.times(); }
const std::vector<double>& FixedTrajectory::masses()        const noexcept { return traj_.masses(); }
const std::vector<std::array<double,3>>& FixedTrajectory::positions()     const noexcept { return traj_.positions(); }
const std::vector<std::array<double,3>>& FixedTrajectory::velocities()    const noexcept { return traj_.velocities(); }
const std::vector<std::array<double,3>>& FixedTrajectory::accelerations() const noexcept { return traj_.accelerations(); }

std::size_t FixedTrajectory::sample_count() const noexcept { return traj_.sample_count(); }
double      FixedTrajectory::mean_dt()      const          { return traj_.mean_dt(); }



LiveTrajectory::LiveTrajectory(double time,
                               double mass,
                               const std::array<double, 3>& position,
                               const std::array<double, 3>& velocity,
                               const std::array<double, 3>& acceleration,
                               double timelimiter,
                               bool finite_timestep,
                               int cadence)
    : time_i(time),
      mass_i(mass),
      position_i(position),
      velocity_i(velocity),
      acceleration_i(acceleration),
      iteration(0),
      limiter(timelimiter),
      require_finite_dt(finite_timestep),
      record_cadence_(cadence)
{
    if (record_cadence_ <= 0) throw std::runtime_error("RecordTrajectoryCadence must be > 0");
    iteration = 0;                                           // initialisation  
    Initial   = OrbitalState{time_i, mass_i, position_i, velocity_i, acceleration_i};      
    //Initial   = OrbitalState{time_i, position_i, velocity_i, acceleration_i};      
    add_event(time, mass, position, velocity, acceleration); // add the initial seed to the timeseries
    //add_event(time, position, velocity, acceleration);
    
}

void LiveTrajectory::add_event(
                double time,
                double mass,
                const std::array<double, 3>& position,
                const std::array<double, 3>& velocity,
                const std::array<double, 3>& acceleration){
    if (!Timeseries.empty() && time < Timeseries.back())
        throw std::runtime_error("Time must be non-decreasing");

    Current = OrbitalState{time, mass, position, velocity, acceleration};

    // We will always append the current event. However, we only want to save 
    // timeseries data every RecordTimeseriesCadence iterations. Thus remove any "last_event"
    // which was not divisible by RecordTimeseriesCadence and not at the beginning of the simulation
    const int last_event       = iteration-1;
    const bool keep_last_event = ((last_event % record_cadence_ == 0) || (last_event<5));
    
    Timeseries.push_back(time);
    Mass.push_back(mass);
    Positions.push_back(position);
    Velocities.push_back(velocity);
    Accelerations.push_back(acceleration);

    if (!keep_last_event) {
        if (Timeseries.size() >= 2) {
            Timeseries.erase(Timeseries.end() - 2);
            Mass.erase(Mass.end() - 2);
            Positions.erase(Positions.end() - 2);
            Velocities.erase(Velocities.end() - 2);
            Accelerations.erase(Accelerations.end() - 2);
            denominators_.erase(denominators_.end() - 1);
        }
    }
    
    // Maintain precomputed denominators for fast interpolation
    if (Timeseries.size() >= 2) {
        double denom = Timeseries.back() - Timeseries[Timeseries.size() - 2];
        denominators_.push_back(denom);
    }
    
    ++iteration;
}

void LiveTrajectory::rebuild_timeseries(
    const std::vector<double>& TrajectoryTimeseries,
    const std::vector<double>& TrajectoryMasses,
    const std::vector<std::array<double, 3>>& TrajectoryPositions,
    const std::vector<std::array<double, 3>>& TrajectoryVelocities,
    const std::vector<std::array<double, 3>>& TrajectoryAccelerations,
    int saved_iteration){
        
    if (saved_iteration < 0)
    throw std::runtime_error("rebuild_timeseries: saved_iteration must be >= 0");

    const std::size_t n = TrajectoryTimeseries.size();
    if (n == 0) throw std::runtime_error("rebuild_timeseries: empty trajectory");
    if (TrajectoryMasses.size() != n || TrajectoryPositions.size() != n || TrajectoryVelocities.size() != n || TrajectoryAccelerations.size() != n)
        throw std::runtime_error("rebuild_timeseries: size mismatch");

    // if (TrajectoryPositions.size() != n || TrajectoryVelocities.size() != n || TrajectoryAccelerations.size() != n)
    //     throw std::runtime_error("rebuild_timeseries: size mismatch");

    if (TrajectoryTimeseries.front() < Initial.t)
        throw std::runtime_error("rebuild_timeseries: first time < Initial.t");

    for (std::size_t i = 1; i < n; ++i) {
        if (TrajectoryTimeseries[i] < TrajectoryTimeseries[i-1])
            throw std::runtime_error("rebuild_timeseries: times must be non-decreasing");}



    Timeseries.clear();
    Mass.clear();
    Positions.clear();
    Velocities.clear();
    Accelerations.clear();

    // Copy vector data
    Timeseries    = TrajectoryTimeseries;
    Mass          = TrajectoryMasses;
    Positions     = TrajectoryPositions;
    Velocities    = TrajectoryVelocities;
    Accelerations = TrajectoryAccelerations;

    // do not rebuild Initial! If we create the class from a checkpoint with *_i known, they will be
    // consistently set in LiveTrajectory::LiveTrajectory!

    //Initial = OrbitalState{Timeseries.front(), Mass.front(),
    //                       Positions.front(), Velocities.front(), Accelerations.front()};

    Current = OrbitalState{Timeseries.back(), Mass.back(), Positions.back(), Velocities.back(), Accelerations.back()};
    //Current = OrbitalState{Timeseries.back(), Positions.back(), Velocities.back(), Accelerations.back()};

    iteration = saved_iteration;
    cached_index_ = 0;  // Reset cache after rebuild
    
    // Precompute all time step denominators for fast interpolation
    denominators_.clear();
    denominators_.reserve(Timeseries.size() - 1);
    for (std::size_t i = 0; i < Timeseries.size() - 1; ++i) {
        denominators_.push_back(Timeseries[i + 1] - Timeseries[i]);
    }
}

double LiveTrajectory::Timestep_dt(const std::array<double, 3>& Force) const {
    // compute timescale mv / F
    double Momentum2 = 0.0;
    double Force2    = 0.0;

    for (std::size_t dim=0; dim<3; ++dim) {
        Momentum2 += (Current.M * Current.V[dim])*(Current.M * Current.V[dim]);
        //Momentum2 += Current.V[dim]*Current.V[dim];
        Force2    += Force[dim] * Force[dim]; 
    }

    if (Force2 == 0.0) {
        if (require_finite_dt) throw std::runtime_error("Undefined dynamical timestep. The force acting on a particle is zero");
        else return std::numeric_limits<double>::infinity();
    }

    return std::sqrt(Momentum2/Force2) * limiter;
}


// efficient interpolation for positions, velocities and accelerations
OrbitalState LiveTrajectory::interpolate(double t) const {
    if (Timeseries.size() < 2) throw std::runtime_error("Need at least 2 samples");
    if (t <= Timeseries.front()) return OrbitalState{t, Mass.front(), Positions.front(), Velocities.front(), Accelerations.front()};
    if (t >= Timeseries.back())  return OrbitalState{t, Mass.back() , Positions.back() , Velocities.back() , Accelerations.back()};
    // if (t <= Timeseries.front()) return OrbitalState{t, Positions.front(), Velocities.front(), Accelerations.front()};
    // if (t >= Timeseries.back())  return OrbitalState{t, Positions.back() , Velocities.back() , Accelerations.back()};


    // Try cached index first (huge win for sequential/nearby queries in bfs3d)
    std::size_t i = cached_index_;
    
    // Ensure i is valid (handle edge case where cached_index_ was at end)
    if (i >= Timeseries.size() - 1) i = Timeseries.size() - 2;
    
    // Check if t is in [Timeseries[i], Timeseries[i+1]]
    bool found = (t >= Timeseries[i] && t <= Timeseries[i + 1]);
    
    // If not found with cached index, try nearby indices first (local search)
    // This is faster than binary search for sequential or nearby queries
    if (!found) {
        // Try i-1 (previous interval)
        if (i > 0 && t >= Timeseries[i-1] && t <= Timeseries[i]) {
            i = i - 1;
            found = true;
        }
        // Try i+2 (next interval after cached)
        else if (i + 2 < Timeseries.size() && t >= Timeseries[i+1] && t <= Timeseries[i+2]) {
            i = i + 1;
            found = true;
        }
    }
    
    // Only fall back to binary search if local search failed
    if (!found) {
        auto it = std::lower_bound(Timeseries.begin(), Timeseries.end(), t);
        i = static_cast<std::size_t>(it - Timeseries.begin()) - 1;
    }
    
    cached_index_ = i;  // Update cache for next call
    
    std::size_t j = i + 1;
    double t0    = Timeseries[i];
    double t1    = Timeseries[j];
    double denom = denominators_[i];  // Use precomputed denominator (O(1) lookup instead of subtraction)

    if (denom == 0.0) {
        return OrbitalState{t, Mass[i], Positions[i], Velocities[i], Accelerations[i]};
        //return OrbitalState{t, Positions[i], Velocities[i], Accelerations[i]};
    }

    double u = (t - t0) / denom;

    OrbitalState out{};
    out.t = t;
    out.M = Mass[i] + u * (Mass[j] - Mass[i]);
    for (std::size_t dim = 0; dim < 3; ++dim) {
        out.X[dim] = Positions[i][dim]     + u * (Positions[j][dim]     - Positions[i][dim]);
        out.V[dim] = Velocities[i][dim]    + u * (Velocities[j][dim]    - Velocities[i][dim]);
        out.A[dim] = Accelerations[i][dim] + u * (Accelerations[j][dim] - Accelerations[i][dim]);
    }
    return out;
}



// struct OrbitalDistribution {
//     int N{};
//     double t{};
//     std::vector<double> M{};
//     std::vector<std::array<int,3>> X{};
//     std::vector<std::array<int,3>> V{};
//     std::vector<std::array<int,3>> A{};
// };

// class Particles {
// public:
//     // initial state
//     int N; 
//     double time_i;

//     // Something to create the distribution?

//     // current state
//     std::vector<double> M{};
//     std::vector<std::array<int,3>> X;
//     std::vector<std::array<int,3>> V;
//     std::vector<std::array<int,3>> A;
    
//     Particles(
//             int Nparticles,
//             double time)
//         : N(Nparticles),
//           time_i(time)
//     {
//         // do something to create initial distribution
//         collect_state(time, mass, position, velocity, acceleration); // add the initial seed to the timeseries
        
//     }

//     void collect_state(
//                     const double time,
//                     const std::vector<double>& M{},
//                     const std::vector<std::array<int,3>>& X,
//                     const std::vector<std::array<int,3>>& V,
//                     const std::vector<std::array<int,3>>& A)
//     {   
//         Current = OrbitalState{time, mass, position, velocity, acceleration};

//         if ((iteration % save_cadence == 0) || ((iteration < 5))){
//             Timeseries.push_back(time);
//             Mass.push_back(mass);
//             Positions.push_back(position);
//             Velocities.push_back(velocity);
//             Accelerations.push_back(acceleration);

//             std::cout << "Record Timeseries\n";
//         }

//         ++iteration;

//     }

// }



// Particles class with many bodies
// Computes global timestep
// Arranges all M, V, A into particles object
// Enables a particular distribution

// Gas calls with certain frequency... 

// Also disable gas for first few timesteps?