#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>

#include "Trajectory.hpp"
#include "utils.hpp"



LiveTrajectory::LiveTrajectory(double time,
                               double mass,
                               const std::array<double, 3>& position,
                               const std::array<double, 3>& velocity,
                               const std::array<double, 3>& acceleration,
                               double timelimiter,
                               bool finite_timestep,
                               double RecordTrajectory_dt)
    : time_i(time),
      mass_i(mass),
      position_i(position),
      velocity_i(velocity),
      acceleration_i(acceleration),
      iteration(0),
      limiter(timelimiter),
      require_finite_dt(finite_timestep),
      RecordTrajectoryCadence(RecordTrajectory_dt)
{
    if (RecordTrajectoryCadence <= 0) throw std::runtime_error("RecordTrajectoryCadence must be > 0");
    inv_RecordTrajectoryCadence = 1.0 / RecordTrajectoryCadence;
    iteration = 0;                                           // initialisation  
    Initial   = OrbitalState{time_i, mass_i, position_i, velocity_i, acceleration_i};      
    add_event(time, mass, position, velocity, acceleration); // add the initial seed to the timeseries
    
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

    // We will always append the current event. However, we only want to save timeseries
    // data every RecordTrajectoryCadence. Thus we perform the following procedure

    // (1 ) append current state 
    Timeseries.push_back(time);
    Mass.push_back(mass);
    Positions.push_back(position);
    Velocities.push_back(velocity);
    Accelerations.push_back(acceleration);

    // (2) Has current state crossed threshold for a new RecordTrajectoryCadence event?
    if (Timeseries.size() >= 2) {
        double tc       = Timeseries[Timeseries.size() - 1];
        double tp       = Timeseries[Timeseries.size() - 2];
        double current  = static_cast<int>(floor((tc-Initial.t) / RecordTrajectoryCadence));
        double previous = static_cast<int>(floor((tp-Initial.t) / RecordTrajectoryCadence));
        bool record_new = (current > previous);
    
        // (3a) If record_new, interpolate at the RecordTrajectoryCadence time and replace previous event
        if (record_new){
            // optional: enforce multiple events per RecordTrajectoryCadence bin
            if (current > previous + 1){ throw std::runtime_error("Trajectory recording cadence is too low");}

            double Mc                 = Mass[Mass.size()-1];
            double Mp                 = Mass[Mass.size()-2];
            double dM                 = Mc - Mp;
            std::array<double, 3>  Xc = Positions[Positions.size() - 1];
            std::array<double, 3>  Xp = Positions[Positions.size() - 2];
            std::array<double, 3>  dX = {Xc[0] - Xp[0], Xc[1] - Xp[1], Xc[2] - Xp[2]};
            std::array<double, 3>  Vc = Velocities[Velocities.size() - 1];
            std::array<double, 3>  Vp = Velocities[Velocities.size() - 2];
            std::array<double, 3>  dV = {Vc[0] - Vp[0], Vc[1] - Vp[1], Vc[2] - Vp[2]};
            std::array<double, 3>  Ac = Accelerations[Accelerations.size() - 1];
            std::array<double, 3>  Ap = Accelerations[Accelerations.size() - 2];
            std::array<double, 3>  dA ={Ac[0] - Ap[0], Ac[1] - Ap[1], Ac[2] - Ap[2]};

            double query_time    = Initial.t + current * RecordTrajectoryCadence;
            double dt            = tc - tp;
            double interval_frac = (query_time - tp) / dt;

            double interp_t                = query_time;
            double interp_M                = Mp + dM * interval_frac;
            std::array<double, 3> interp_X = {Xp[0] + interval_frac * dX[0], Xp[1] + interval_frac * dX[1], Xp[2] + interval_frac * dX[2]};
            std::array<double, 3> interp_V = {Vp[0] + interval_frac * dV[0], Vp[1] + interval_frac * dV[1], Vp[2] + interval_frac * dV[2]};
            std::array<double, 3> interp_A = {Ap[0] + interval_frac * dA[0], Ap[1] + interval_frac * dA[1], Ap[2] + interval_frac * dA[2]};

            // overwrite the second last value with the interpolated value
            Timeseries[Timeseries.size() - 2]       = interp_t;
            Mass[Mass.size() - 2]                   = interp_M;
            Positions[Positions.size() - 2]         = interp_X;
            Velocities[Velocities.size() - 2]       = interp_V;
            Accelerations[Accelerations.size() - 2] = interp_A;
        }
        else{
            // (3b) still in same interval, just remove outdated event
            Timeseries.erase(Timeseries.end() - 2);
            Mass.erase(Mass.end() - 2);
            Positions.erase(Positions.end() - 2);
            Velocities.erase(Velocities.end() - 2);
            Accelerations.erase(Accelerations.end() - 2);
        }
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

    // 4) copy exactly
    Timeseries    = TrajectoryTimeseries;
    Mass          = TrajectoryMasses;
    Positions     = TrajectoryPositions;
    Velocities    = TrajectoryVelocities;
    Accelerations = TrajectoryAccelerations;

    // do not rebuild Initial! If we create the class from a checkpoint with *_i known, they will be
    // consistently set in LiveTrajectory::LiveTrajectory!

    Current = OrbitalState{Timeseries.back(), Mass.back(),
                           Positions.back(), Velocities.back(), Accelerations.back()};

    iteration = saved_iteration;
    cached_index = 0;  // Reset interpolation cache

}

double LiveTrajectory::Timestep_dt(const std::array<double, 3>& Force) const {
    // compute timescale mv / F
    double Momentum2 = 0.0;
    double Force2    = 0.0;

    for (std::size_t dim=0; dim<3; ++dim) {
        Momentum2 += (Current.M * Current.V[dim])*(Current.M * Current.V[dim]);
        Force2    += Force[dim] * Force[dim];
    }

    if (Force2 == 0.0) {
        if (require_finite_dt) throw std::runtime_error("Undefined dynamical timestep. The force acting on a particle is zero");
        else return std::numeric_limits<double>::infinity();
    }

    return std::sqrt(Momentum2/Force2) * limiter;
}


// inline void LiveTrajectory::interpolate_into(
//     double t,
//     double& M_out,
//     std::array<double,3>& X_out,
//     std::array<double,3>& V_out,
//     std::array<double,3>& A_out) const
// {
//     const std::size_t N = Timeseries.size();

// #ifndef NDEBUG
//     if (N < 2) throw std::runtime_error("Need at least 2 samples");
//     if (Positions.size() != N || Mass.size() != N || Velocities.size() != N || Accelerations.size() != N)
//         throw std::runtime_error("Series sizes mismatch");
// #endif

//     // Endpoint clamps (fast path)
//     if (t <= Timeseries.front()) {
//         M_out = Mass.front();
//         X_out = Positions.front();
//         V_out = Velocities.front();
//         A_out = Accelerations.front();
//         return;
//     }
//     if (t >= Timeseries.back()) {
//         M_out = Mass.back();
//         X_out = Positions.back();
//         V_out = Velocities.back();
//         A_out = Accelerations.back();
//         return;
//     }

//     // Determine bracket index i (low), j=i+1 (high)
//     const double time_reg = Timeseries[N - 2];
//     int i;

//     if (t > time_reg) {
//         // Tail segment: interpolate between last regular sample (N-2) and live sample (N-1)
//         i = (int)N - 2;
//     } else {
//         // Regular segment: O(1) index by uniform cadence aligned to Initial.t
//         // Use multiply by cached reciprocal (faster than divide + floor in many cases).
//         const double x = (t - Initial.t) * inv_RecordTrajectoryCadence;

//         // We want floor(x). Truncation is ok for x>=0 (which it is here due to endpoint check),
//         // but handle any tiny negative due to floating-point roundoff.
//         int ii = (int)x;
//         if ((double)ii > x) --ii;

//         if (ii < 0) ii = 0;

//         // Clamp to ensure ii+1 exists and stays within regular region.
//         // Regular region runs up to N-2, so ii must be <= N-3 here.
//         const int max_i = (int)N - 3;
//         if (ii > max_i) ii = max_i;

//         i = ii;
//     }

//     const int j = i + 1;

//     const double t0 = Timeseries[(std::size_t)i];
//     const double t1 = Timeseries[(std::size_t)j];
//     const double denom = (t1 - t0);
//     const double u = (denom != 0.0) ? (t - t0) / denom : 0.0;

//     // Scalars
//     const double M0 = Mass[(std::size_t)i];
//     const double M1 = Mass[(std::size_t)j];
//     M_out = M0 + u * (M1 - M0);

//     // References to avoid copying std::array<double,3>
//     const auto& X0 = Positions[(std::size_t)i];
//     const auto& X1 = Positions[(std::size_t)j];
//     const auto& V0 = Velocities[(std::size_t)i];
//     const auto& V1 = Velocities[(std::size_t)j];
//     const auto& A0 = Accelerations[(std::size_t)i];
//     const auto& A1 = Accelerations[(std::size_t)j];

//     // Direct interpolation (no dX/dV/dA temporaries)
//     for (int d = 0; d < 3; ++d) {
//         X_out[d] = X0[d] + u * (X1[d] - X0[d]);
//         V_out[d] = V0[d] + u * (V1[d] - V0[d]);
//         A_out[d] = A0[d] + u * (A1[d] - A0[d]);
//     }
// }

// efficient interpolation for positions, velocities and accelerations
OrbitalState LiveTrajectory::interpolate(double t) const {

    static std::atomic<long long> n_calls{0};
    ++n_calls;
    if ((n_calls % 100000000LL) == 0) std::cerr << "interpolate calls: " << n_calls << "\n";


    if (Timeseries.size() < 2) throw std::runtime_error("Need at least 2 samples");
    if (t <= Timeseries.front()) return OrbitalState{t, Mass.front(), Positions.front(), Velocities.front(), Accelerations.front()};
    if (t >= Timeseries.back())  return OrbitalState{t, Mass.back() , Positions.back() , Velocities.back() , Accelerations.back()};

    if (Positions.size() != Timeseries.size() ||
        Mass.size() != Timeseries.size() ||
        Velocities.size() != Timeseries.size() ||
        Accelerations.size() != Timeseries.size())
        throw std::runtime_error("Series sizes mismatch");

    
    double time_reg = Timeseries[Timeseries.size() - 2];
    int bracket_low;
    if (t > time_reg){
        bracket_low = Timeseries.size() - 2;        
    }
    else{
        bracket_low = static_cast<int>(floor((t-Initial.t) / RecordTrajectoryCadence));
    }

    if ((bracket_low < 0) || (bracket_low >= (int)Timeseries.size()-1)) throw std::runtime_error("Interpolation Error. Lower bracket has out of bounds index");


    int bracket_high     = bracket_low + 1;
    double interval_frac = (t - Timeseries[bracket_low]) / (Timeseries[bracket_high] - Timeseries[bracket_low]);

    double Ml                 = Mass[bracket_low];
    double Mh                 = Mass[bracket_high];
    double dM                 = Mh - Ml;

    // use references instead of copying
    const auto& Xl = Positions[bracket_low];
    const auto& Xh = Positions[bracket_high];
    const auto& Vl = Velocities[bracket_low];
    const auto& Vh = Velocities[bracket_high];
    const auto& Al = Accelerations[bracket_low];
    const auto& Ah = Accelerations[bracket_high];

    std::array<double, 3>  dX = {Xh[0] - Xl[0], Xh[1] - Xl[1], Xh[2] - Xl[2]};
    std::array<double, 3>  dV = {Vh[0] - Vl[0], Vh[1] - Vl[1], Vh[2] - Vl[2]};
    std::array<double, 3>  dA ={Ah[0] - Al[0], Ah[1] - Al[1], Ah[2] - Al[2]};

    double interp_M                = Ml + dM * interval_frac;
    std::array<double, 3> interp_X = {Xl[0] + interval_frac * dX[0], Xl[1] + interval_frac * dX[1], Xl[2] + interval_frac * dX[2]};
    std::array<double, 3> interp_V = {Vl[0] + interval_frac * dV[0], Vl[1] + interval_frac * dV[1], Vl[2] + interval_frac * dV[2]};
    std::array<double, 3> interp_A = {Al[0] + interval_frac * dA[0], Al[1] + interval_frac * dA[1], Al[2] + interval_frac * dA[2]};

    return OrbitalState{t, interp_M, interp_X, interp_V, interp_A};
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