#include <vector>
#include <tuple>
#include <queue>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstddef>
#include <stdexcept>
#include <array>
#include <cstdint>


#include "utils.hpp"
#include "Trajectory.hpp"
#include "RootFinder.hpp"
#include "GaseousDynamics.hpp"




// ===============================================================
// Propagate function 
// ===============================================================
bool propagate_full(
    double tr,
    double t,
    const std::array<double,3>& X,
    const std::array<double,3>& dX,
    double cs,
    double error_tol,
    const LiveTrajectory& traj,
    double* tau_out,
    double* err_out){

    // 0. Initialise
    double X_rel[3];
    double Projection[3];
    double PV=0;
    double PA=0;
    double Px=0;
    double Ax=0;
    double Vx=0;
    double xx=0;
    double VV=0;

    if (!std::isfinite(tr)) return false;
    constexpr int dim = 3;

    // 2. Predictor Step
    double M_r;
    std::array<double,3> X_r, V_r, A_r;
    traj.interpolate_into(tr, M_r, X_r, V_r, A_r);


    for (int d = 0; d < dim; ++d) X_rel[d] = X_r[d] - X[d];
    double X_rel_norm = std::sqrt(X_rel[0]*X_rel[0] + X_rel[1]*X_rel[1] + X_rel[2]*X_rel[2]);
    if (X_rel_norm < 1e-15) return false;

    for (int d = 0; d < dim; ++d) {
        Projection[d] = X_rel[d]/X_rel_norm;
        PV           += Projection[d]*V_r[d];
        PA           += Projection[d]*A_r[d];
        Px           += Projection[d]*dX[d];
        Ax           += A_r[d]*dX[d];
        Vx           += V_r[d]*dX[d];
        xx           += dX[d]*dX[d];
        VV           += V_r[d]*V_r[d];
    }

    double Constant              = -Px + (xx - Px*Px) / (2 * X_rel_norm);
    double Linear_Coefficient    = cs + PV + (PV*Px - Vx)/X_rel_norm;
    double Quadratic_Coefficient = 0.5*PA + (VV - Ax - PV*PV + PA*Px)/(2*X_rel_norm);
    double D                     = Linear_Coefficient*Linear_Coefficient - 4.0*Quadratic_Coefficient*Constant;
    double dt_pred;
    if (D < 0.0) {
        dt_pred = -Constant / Linear_Coefficient;
    } else {
        double sqrtD  = std::sqrt(D);
        double q      = -0.5 * (Linear_Coefficient + std::copysign(sqrtD, Linear_Coefficient));
        double r1     = q / Quadratic_Coefficient;
        double r2     = Constant / q;
        double dt_lin = -Constant / Linear_Coefficient;
        dt_pred       = (std::abs(r1 - dt_lin) < std::abs(r2 - dt_lin)) ? r1 : r2;
    }

    double tau = tr + dt_pred;
    double X_new[3] = { X[0]+dX[0], X[1]+dX[1], X[2]+dX[2] };

    // 3. Halley Correction Step
    double M_tau;
    std::array<double,3> X_tau, V_tau, A_tau;
    traj.interpolate_into(tau, M_tau, X_tau, V_tau, A_tau);


    double r_vec[3]   = { X_tau[0]-X_new[0], X_tau[1]-X_new[1], X_tau[2]-X_new[2] };
    double R          = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    if (R < 1e-15) return false;
    double dist       = R;
    double first_err  = std::abs((tau - t) + dist / cs);
    double n[3]       = { r_vec[0]/R, r_vec[1]/R, r_vec[2]/R };
    double v_dot_n    = n[0]*V_tau[0] + n[1]*V_tau[1] + n[2]*V_tau[2];
    double F          = (tau - t) + R / cs;
    double Fp         = 1.0 + v_dot_n / cs;
    if (std::abs(Fp) < 1e-15) return false;
    double v_sq       = V_tau[0]*V_tau[0] + V_tau[1]*V_tau[1] + V_tau[2]*V_tau[2];
    double n_dot_a    = n[0]*A_tau[0] + n[1]*A_tau[1] + n[2]*A_tau[2];
    double Fpp        = (n_dot_a + (v_sq - v_dot_n*v_dot_n)/R) / cs;
    double denom      = 2*Fp*Fp - F*Fpp;

    if (std::abs(denom) > 1e-20)
        tau -= (2*F*Fp)/denom;    // Halley's method
    else
        tau -= F/Fp;              // fallback to Newton's method

    // 3. Define error with refined tau
    traj.interpolate_into(tau, M_tau, X_tau, V_tau, A_tau);

    
    r_vec[0]   = X_tau[0]-X_new[0];
    r_vec[1]   = X_tau[1]-X_new[1];
    r_vec[2]   = X_tau[2]-X_new[2];
    R           = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    if (R < 1e-15) return false;
    dist        = R;
    n[0]        = r_vec[0]/R;
    n[1]        = r_vec[1]/R;
    n[2]        = r_vec[2]/R;
    v_dot_n    = n[0]*V_tau[0] + n[1]*V_tau[1] + n[2]*V_tau[2];
    F          = (tau - t) + R / cs;
    Fp         = 1.0 + v_dot_n / cs;
    if (std::abs(Fp) < 1e-15) return false;
    v_sq       = V_tau[0]*V_tau[0] + V_tau[1]*V_tau[1] + V_tau[2]*V_tau[2];
    n_dot_a    = n[0]*A_tau[0] + n[1]*A_tau[1] + n[2]*A_tau[2];
    Fpp        = (n_dot_a + (v_sq - v_dot_n*v_dot_n)/R) / cs;

    if (tau < traj.initial().t || tau > traj.current().t) {
        *tau_out = NAN;
        *err_out = NAN;
        return false;
    }

    double second_err  = std::abs((tau - t) + dist / cs);
    double err;
    if (first_err<second_err){
        err = first_err;
        tau = tr + dt_pred;
        }
    else
        err = second_err;

    *tau_out = (err > error_tol ? NAN : tau);
    *err_out = err;
    return true;
}





// ===============================================================
// bfs3d (low-churn, flattened grid, per-slot queue + stale-skip)
// ===============================================================
//
// Key changes vs your original:
//  1) Flattened voxel index (idx = (i*Ny + j)*Nz + k) for fast queue + compact per-slot arrays.
//  2) Queue items are (idx, slot, err_snapshot). On pop we skip stale entries -> huge churn reduction.
//  3) Optional in_queue flag prevents queue bloat (still correct with stale-skip).
//  4) "Significant improvement" gate avoids re-enqueueing on tiny numerical improvements.
//  5) Branching is conservative (primary failure/out-of-tol only) and tries -eps only if +eps didn't produce a valid insert.
//
// NOTE: This keeps your multi-root logic (MaxRoots slots per voxel) and preserves correctness
// under your replace/update policy. It does NOT use a voxel-level visited[].
//
// You can paste this in place of your existing bfs3d. It still uses:
//   - Mat4D Roots, Errors with operator()(i,j,k,s)
//   - Mat2I Neighbours with Neighbours.N and Neighbours(n,0/1/2)
//   - propagate_full(...)
//   - traj.mean_dt()
//

void bfs3d(
    Mat4D& Roots,
    Mat4D& Errors,
    const std::vector<double>& X,
    const std::vector<double>& Y,
    const std::vector<double>& Z,
    const Mat2I& Neighbours,
    double t,
    double sound_speed,
    double error_tol,
    double unique_tol,
    const LiveTrajectory& traj)
{
    const int Nx       = Roots.Nx;
    const int Ny       = Roots.Ny;
    const int Nz       = Roots.Nz;
    const int MaxRoots = Roots.Nr;

    const size_t total_cells = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    const size_t total_slots = total_cells * (size_t)MaxRoots;

    std::vector<double> best_err(total_slots, std::numeric_limits<double>::infinity());

    // Branch perturbation
    const double dt_grid = traj.mean_dt();
    const double eps_tr  = (dt_grid != 0.0) ? (0.25 * dt_grid) : 1e-6;

    // -----------------------------
    // Flatten / Unflatten helpers
    // -----------------------------
    const size_t NyNz = (size_t)Ny * (size_t)Nz;

    auto flatten = [&](int i, int j, int k) -> uint32_t {
        return (uint32_t)(((size_t)i * (size_t)Ny + (size_t)j) * (size_t)Nz + (size_t)k);
    };

    auto unflatten = [&](uint32_t idx, int& i, int& j, int& k) {
        size_t sidx = (size_t)idx;
        i = (int)(sidx / NyNz);
        size_t rem = sidx - (size_t)i * NyNz;
        j = (int)(rem / (size_t)Nz);
        k = (int)(rem - (size_t)j * (size_t)Nz);
    };

    auto id_slot = [&](uint32_t idx, int s) -> size_t {
        return (size_t)idx * (size_t)MaxRoots + (size_t)s;
    };

    // -----------------------------
    // Anti-churn parameters
    // -----------------------------
    // Only accept/enqueue when improvement is "real" (prevents micro-oscillation churn).
    const double improve_abs = 1e-14;
    const double improve_rel = 1e-6;

    auto is_significant_improvement = [&](double new_err, double old_err) -> bool {
        if (!std::isfinite(old_err)) return true;
        const double thresh = improve_abs + improve_rel * std::abs(old_err);
        return (new_err < old_err - thresh);
    };

    // -----------------------------
    // Queue node with stale-skip
    // -----------------------------
    struct Node {
        uint32_t idx;        // flattened voxel index
        uint8_t  slot;       // 0..MaxRoots-1
        float    err_snap;   // snapshot of best_err at enqueue time (float usually enough)
    };

    std::deque<Node> q;

    // Optional queue-bloat control: only keep at most one pending entry per (idx,slot).
    // Correctness is preserved because best_err holds the true best label; stale-skip handles updates.
    std::vector<uint8_t> in_queue(total_slots, 0);

    auto try_enqueue = [&](uint32_t idx, int slot, double err) {
        const size_t id = id_slot(idx, slot);
        if (in_queue[id]) return; // already pending; best_err will be processed when that node pops
        in_queue[id] = 1;
        q.push_back(Node{idx, (uint8_t)slot, (float)err});
    };

    // -----------------------------
    // Insert/update candidate into voxel (i,j,k).
    // Policy:
    //  1) match within unique_tol -> update if significantly better
    //  2) else insert into empty slot if available
    //  3) else replace worst slot if significantly better
    // On commit, update Roots/Errors/best_err and enqueue (idx,slot).
    // -----------------------------
    auto insert_candidate = [&](int i, int j, int k, uint32_t idx, double tau, double err) -> int {
        if (!std::isfinite(tau) || !std::isfinite(err)) return -1;
        if (err > error_tol) return -1;

        int match_slot = -1;
        int empty_slot = -1;

        for (int s = 0; s < MaxRoots; ++s) {
            const double ex = Roots(i,j,k,s);
            if (!std::isfinite(ex)) {
                if (empty_slot < 0) empty_slot = s;
                continue;
            }
            if (std::abs(tau - ex) < unique_tol) {
                match_slot = s;
                break;
            }
        }

        auto commit = [&](int s) {
            const size_t id = id_slot(idx, s);
            Roots(i,j,k,s)  = tau;
            Errors(i,j,k,s) = err;
            best_err[id]    = err;
            try_enqueue(idx, s, err);
        };

        if (match_slot >= 0) {
            const size_t id = id_slot(idx, match_slot);
            const double old = best_err[id];
            if (is_significant_improvement(err, old)) commit(match_slot);
            return match_slot;
        }

        if (empty_slot >= 0) {
            const size_t id = id_slot(idx, empty_slot);
            const double old = best_err[id];
            if (is_significant_improvement(err, old)) {
                commit(empty_slot);
                return empty_slot;
            }
            return -1;
        }

        int worst_slot = 0;
        double worst_e = -std::numeric_limits<double>::infinity();
        for (int s = 0; s < MaxRoots; ++s) {
            const double e = best_err[id_slot(idx, s)];
            if (e > worst_e) { worst_e = e; worst_slot = s; }
        }

        if (is_significant_improvement(err, worst_e)) {
            commit(worst_slot);
            return worst_slot;
        }

        return -1;
    };

    // -----------------------------
    // Seed queue from existing finite roots
    // -----------------------------
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const uint32_t idx = flatten(i,j,k);
                for (int s = 0; s < MaxRoots; ++s) {
                    const double tau0 = Roots(i,j,k,s);
                    if (!std::isfinite(tau0)) continue;

                    double err0 = Errors(i,j,k,s);
                    if (!std::isfinite(err0)) err0 = 0.0;

                    const size_t id = id_slot(idx, s);
                    if (is_significant_improvement(err0, best_err[id])) {
                        best_err[id] = err0;
                        try_enqueue(idx, s, err0);
                    }
                }
            }
        }
    }

    // Timing (your original reporting style)
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t counter  = 0;
    size_t print_interval = 1000000;

    // Branch control: conservative to avoid huge extra work.
    // If you later expose conditioning from propagate_full (|Fp|, denom), branch on that instead.
    constexpr bool branch_on_failure_only = true;

    // -----------------------------
    // Main loop
    // -----------------------------
    while (!q.empty()) {
        const Node node = q.front();
        q.pop_front();

        const size_t here_id = id_slot(node.idx, (int)node.slot);

        // Clear in_queue for this slot now that we're handling it
        // (we may re-enqueue later if an improvement happens)
        in_queue[here_id] = 0;

        // Stale-skip (major churn killer)
        if ((double)node.err_snap != best_err[here_id]) continue;

        int i, j, k;
        unflatten(node.idx, i, j, k);

        const double tr_here = Roots(i,j,k,(int)node.slot);
        if (!std::isfinite(tr_here)) continue;

        const std::array<double,3> X_here = { X[(size_t)i], Y[(size_t)j], Z[(size_t)k] };

        // Expand to neighbors
        for (int n = 0; n < Neighbours.N; ++n) {
            const int di = (int)Neighbours(n,0);
            const int dj = (int)Neighbours(n,1);
            const int dk = (int)Neighbours(n,2);

            const int ni = i + di;
            const int nj = j + dj;
            const int nk = k + dk;

            if (ni < 0 || nj < 0 || nk < 0 || ni >= Nx || nj >= Ny || nk >= Nz)
                continue;

            const uint32_t nidx = flatten(ni, nj, nk);

            const std::array<double,3> dX = {
                X[(size_t)ni] - X[(size_t)i],
                Y[(size_t)nj] - Y[(size_t)j],
                Z[(size_t)nk] - Z[(size_t)k]
            };

            // 1) Primary propagation
            double tauA = NAN, errA = NAN;
            const bool okA = propagate_full(
                tr_here, t, X_here, dX,
                sound_speed, error_tol,
                traj, &tauA, &errA
            );

            bool primary_valid = (okA && std::isfinite(tauA) && std::isfinite(errA) && errA <= error_tol);
            bool insertedA = false;

            if (primary_valid) {
                insertedA = (insert_candidate(ni, nj, nk, nidx, tauA, errA) >= 0);
            }

            // 2) Branching (conservative to avoid churn)
            bool need_branch = false;
            if (branch_on_failure_only) {
                need_branch = !primary_valid;
            } else {
                need_branch = !(primary_valid && insertedA);
            }

            if (need_branch) {
                // Try +eps
                double tauB = NAN, errB = NAN;
                const bool okB = propagate_full(
                    tr_here + eps_tr, t, X_here, dX,
                    sound_speed, error_tol,
                    traj, &tauB, &errB
                );

                bool b_valid = (okB && std::isfinite(tauB) && std::isfinite(errB) && errB <= error_tol);
                bool insertedB = false;
                if (b_valid) {
                    insertedB = (insert_candidate(ni, nj, nk, nidx, tauB, errB) >= 0);
                }

                // Try -eps only if +eps didn't yield a usable insert (saves work)
                if (!insertedB) {
                    double tauC = NAN, errC = NAN;
                    const bool okC = propagate_full(
                        tr_here - eps_tr, t, X_here, dX,
                        sound_speed, error_tol,
                        traj, &tauC, &errC
                    );

                    if (okC && std::isfinite(tauC) && std::isfinite(errC) && errC <= error_tol) {
                        insert_candidate(ni, nj, nk, nidx, tauC, errC);
                    }
                }
            }

            counter++;
            if (counter % print_interval == 0) {
                const auto now       = std::chrono::high_resolution_clock::now();
                const double elapsed = std::chrono::duration<double>(now - start_time).count();
                const double rate    = counter / elapsed;
                std::cerr << "[bfs3d] " << std::setw(10) << std::setfill('0') << counter
                          << " propagations at rate " << rate/1000000 << " Mzps\n";
            }
        }
    }
}




// ===============================================================
// Translate roots to density perturbations
// ===============================================================

void Alpha3d(
    Mat3D& Alpha,
    Mat3D& NRoots,
    const Mat4D& Roots,
    const std::vector<double>& X,
    const std::vector<double>& Y,
    const std::vector<double>& Z,
    const LiveTrajectory& traj,
    const double cs)
{

    // double xp0          = t_arr[0];
    // double dx           = t_arr[1] - t_arr[0];
    // const int nt        = t_arr.size();
    // int dim             = 3;
    // const double* x_ptr = x_arr.data();
    // const double* v_ptr = v_arr.data();

    const double NonLinearParameter = G * traj.current().M / cs / cs;

    const int Nx       = Roots.Nx;
    const int Ny       = Roots.Ny;
    const int Nz       = Roots.Nz;
    const int MaxRoots = Roots.Nr;


    //Mat3D Alpha(Nx, Ny, Nz);
    //Mat3D NRoots(Nx, Ny, Nz);

    const double tol = 5e-7;

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k) {

            double rootweight = 0.0;
            int rootnumber    = 0;

            // Temporary storage of unique roots for this voxel
            std::vector<double> unique_roots;
            unique_roots.reserve(MaxRoots);

            for (int r = 0; r < MaxRoots; ++r) {
                double tr = Roots(i, j, k, r);
                if (!std::isfinite(tr))
                    continue;

                // --- duplicate check ---
                bool duplicate = false;
                for (double existing : unique_roots) {
                    if (std::abs(tr - existing) < tol) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate)
                    continue;
                // ------------------------

                // add to unique list
                unique_roots.push_back(tr);

                // --- your existing physics calculations ---
                double x = X[i];
                double y = Y[j];
                double z = Z[k];

                OrbitalState retarded_state = traj.interpolate(tr);
                std::array<double,3> X_r    = retarded_state.X;
                std::array<double,3> V_r    = retarded_state.V;
                // double X_r[3], V_r[3];
                // interp_linear_uniform(xp0, dx, x_ptr, nt, dim, tr, X_r);
                // interp_linear_uniform(xp0, dx, v_ptr, nt, dim, tr, V_r);

                double X_rel[3] = {X_r[0] - x, X_r[1] - y, X_r[2] - z};
                double X_rel_norm = std::sqrt(
                    X_rel[0]*X_rel[0] + X_rel[1]*X_rel[1] + X_rel[2]*X_rel[2]);

                double denom = std::abs(X_rel_norm +
                    (X_rel[0]*V_r[0] + X_rel[1]*V_r[1] + X_rel[2]*V_r[2]) / cs);

                rootweight += 1.0 / denom;
                rootnumber += 1;
            }

            Alpha(i,j,k)  = rootweight * NonLinearParameter;
            NRoots(i,j,k) = rootnumber;
        }
    //return {Alpha, NRoots};
}


// ===============================================================
// Assemble total density wake from all particles?
// ===============================================================


// ===============================================================
// Integrate density wake to get force from a given position
// ===============================================================

std::array<double,3> ComputeGasForce(
    const OrbitalState& p,
    const SpatialDomain& domain,
    const Mat3D& Alpha,
    double rho_0,
    double rmin)
{
    const std::size_t Nx = domain.Resolution_x;
    const std::size_t Ny = domain.Resolution_y;
    const std::size_t Nz = domain.Resolution_z;

    // particle mass
    const double M = p.M;

    // particle position
    const double px = p.X[0];
    const double py = p.X[1];
    const double pz = p.X[2];

    // assuming uniform grid spacing 
    const double dx = (Nx > 1) ? (domain.X[1] - domain.X[0]) : 0.0;
    const double dy = (Ny > 1) ? (domain.Y[1] - domain.Y[0]) : 0.0;
    const double dz = (Nz > 1) ? (domain.Z[1] - domain.Z[0]) : 0.0;
    const double dV = dx * dy * dz;

    std::array<double,3> F{0.0, 0.0, 0.0};

    for (std::size_t i = 0; i < Nx; ++i) {
        const double x = domain.X[i];
        const double rx = x - px;

        for (std::size_t j = 0; j < Ny; ++j) {
            const double y = domain.Y[j];
            const double ry = y - py;

            for (std::size_t k  = 0; k < Nz; ++k) {
                const double z  = domain.Z[k];
                const double rz = z - pz;

                const double a  = Alpha(i,j,k);
                const double r2 = rx*rx + ry*ry + rz*rz;
                const double r  = std::sqrt(r2);

                if (r <= rmin) continue;

                const double r3 = r2 * r;
                const double coeff = (-G * M * rho_0 * a * dV) / r3;

                F[0] += coeff * rx;
                F[1] += coeff * ry;
                F[2] += coeff * rz;
            }
        }
    }
    return F;
}





// void RK2_ParticleIntegrator(
//     std::vector<Particle>& particles,
//     double dt,
//     double t0){
//     // Step 1: compute acceleration a0(x, v, t0)
//     ComputeGasAcceleration(particles, t0); // Careful. Do we want for a given particle or all of them?

//     // Save v_n and build v_mid = v_n + (dt/2)*a0
//     const double half = 0.5 * dt;
//     std::vector<std::array<double,3>> v0(particles.size());
//     for (std::size_t i = 0; i < particles.size(); ++i) {
//         v0[i] = particles[i].v;
//         particles[i].v[0] = v0[i][0] + half * particles[i].a[0];
//         particles[i].v[1] = v0[i][1] + half * particles[i].a[1];
//         particles[i].v[2] = v0[i][2] + half * particles[i].a[2];
//     }

//     // Stage 2: a_mid = a_gas(x, v_mid, t0 + dt/2)
//     ComputeGasAcceleration(particles, t0 + half);

//     // Final update: v_{n+1} = v_n + dt * a_mid
//     for (std::size_t i = 0; i < particles.size(); ++i) {
//         particles[i].v[0] = v0[i][0] + dt * particles[i].a[0];
//         particles[i].v[1] = v0[i][1] + dt * particles[i].a[1];
//         particles[i].v[2] = v0[i][2] + dt * particles[i].a[2];
//     }
// }



LinearGasSolver::LinearGasSolver(
    const LiveTrajectory& traj,
    const SpatialDomain& domain,
    double cs,
    int Nroots,
    double unique_tolerance,
    double error_tolerance,
    double rho,
    double rminimum): 
        Trajectory(traj), 
        Domain(domain), 
        sound_speed(cs), 
        rmin(rminimum), 
        unique_tol(unique_tolerance), 
        error_tol(error_tolerance), 
        rho_0(rho), 
        Max_Number_of_Roots(Nroots), 
        Roots((int)domain.Resolution_x, (int)domain.Resolution_y, (int)domain.Resolution_z, Nroots), 
        Errors((int)domain.Resolution_x, (int)domain.Resolution_y, (int)domain.Resolution_z, Nroots), 
        Alpha((int)domain.Resolution_x, (int)domain.Resolution_y, (int)domain.Resolution_z), 
        NRoots((int)domain.Resolution_x, (int)domain.Resolution_y, (int)domain.Resolution_z)
{}


double LinearGasSolver::RootFunction(double tr, double t, const std::array<double,3> X)
{
    OrbitalState retarded_state = Trajectory.interpolate(tr);
    std::array<double,3> X_r    = retarded_state.X;
    std::array<double,3> dX{ X_r[0]-X[0], X_r[1]-X[1], X_r[2]-X[2]};
    double dist                 = std::sqrt(dX[0]*dX[0] + dX[1]*dX[1] + dX[2]*dX[2]);
    return tr - t + dist / sound_speed;
}


std::vector<double> LinearGasSolver::RootValues(
    double t, 
    const std::array<double,3> X,
    double a,
    double b){

        if (!(a < b)) return {};

        double BrentTol  = 1e-12;
        double BrentIter = 100;
        double Tol       = 1e-5;
        double dx        = 1e-4;
        double Offset    = 1e-3;
        double MaxIter   = 2000;
        double rhobeg    = 0.01;

        double f_a       = RootFunction(a, t, X);
        double f_b       = RootFunction(b, t, X);


        auto f = [&](double tr) { return RootFunction(tr, t, X); };
        GodTierRootFinder rf(
            f,
            BrentTol,
            BrentIter,
            Tol,
            dx,
            Offset,
            MaxIter,
            rhobeg,
            Max_Number_of_Roots
        );
        return rf.Roots(a, b, f_a, f_b);
}


void LinearGasSolver::ComputeSeedRoots(double t){
    // Set Rooots to nan everywhere
    using clock = std::chrono::high_resolution_clock;

    const auto t_start = clock::now();

    std::size_t root_calls = 0;

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i=0; i<Domain.Resolution_x; ++i)
    for (std::size_t j=0; j<Domain.Resolution_y; ++j)
    for (std::size_t k=0; k<Domain.Resolution_z; ++k){      
        Alpha(i,j,k) = 0.0;
        for (std::size_t r=0; r<Max_Number_of_Roots; ++r){
            Roots(i,j,k,r)  = NaN;
            Errors(i,j,k,r) = NaN;
        }
    }

    for (const auto& origin : Domain.SeedOrigins) {
        int ix = origin[0], iy = origin[1], iz = origin[2];
        std::array<double,3> X    = { Domain.X[ix], Domain.Y[iy], Domain.Z[iz] };
        std::vector<double> roots = RootValues(t, X, Trajectory.initial().t, Trajectory.current().t);

        // fill up valid seeds
        int n = std::min<int>((int)roots.size(), Max_Number_of_Roots);
        for (int r = 0; r < n; ++r) {
            Roots(ix,iy,iz,r)  = roots[r];
            Errors(ix,iy,iz,r) = 0.0;
        }
        // render the rest as invalid
        for (int r = n; r < Max_Number_of_Roots; ++r) {
            Roots(ix,iy,iz,r)  = NaN;
            Errors(ix,iy,iz,r) = NaN;
        }
    ++root_calls;
    }

    const auto t_end = clock::now();
    const double seconds =
        std::chrono::duration<double>(t_end - t_start).count();

    const double rate = root_calls / seconds / 1000000;

    std::cerr << "[Seeds] Computed at rate"
              << rate << " Mzps\n";
}


void LinearGasSolver::compute_wake(double t){
    
    // 1. Compute the seeds at requested time
    ComputeSeedRoots(t);

    // 2. Define the neighbour points for each voxel:
    Mat2I Neighbours(6, 3);

    Neighbours(0,0) =  1;  Neighbours(0,1) = 0;  Neighbours(0,2) = 0;
    Neighbours(1,0) = -1;  Neighbours(1,1) = 0;  Neighbours(1,2) = 0;
    Neighbours(2,0) =  0;  Neighbours(2,1) = 1;  Neighbours(2,2) = 0;
    Neighbours(3,0) =  0;  Neighbours(3,1) = -1; Neighbours(3,2) = 0;
    Neighbours(4,0) =  0;  Neighbours(4,1) = 0;  Neighbours(4,2) = 1;
    Neighbours(5,0) =  0;  Neighbours(5,1) = 0;  Neighbours(5,2) = -1;


    // 2. launch the BFS algorithm
    bfs3d(
        Roots,
        Errors,
        Domain.X,
        Domain.Y,
        Domain.Z,
        Neighbours,              
        t,
        sound_speed,
        error_tol,
        unique_tol,
        Trajectory
    );

    // 3. Construct the density wake Alpha
    Alpha3d(
        Alpha,
        NRoots,
        Roots,
        Domain.X,
        Domain.Y,
        Domain.Z,
        Trajectory,
        sound_speed
    );

    // 4. Compute gravitational force
    OrbitalState p = Trajectory.current();
    CurrentForce   = ComputeGasForce(
        p,
        Domain,
        Alpha,
        rho_0,
        rmin
    );

    // 5. Save Force to the Solver
    ForceSeries.push_back(CurrentForce);
    ForceTimes.push_back(t);
}

    