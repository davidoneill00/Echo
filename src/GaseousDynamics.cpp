#include <vector>
#include <tuple>
#include <queue>
#include <set>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstddef>
#include <stdexcept>
#include <array>
#include <algorithm>


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
    OrbitalState retarded_state = traj.interpolate(tr);
    std::array<double,3> X_r = retarded_state.X;
    std::array<double,3> V_r = retarded_state.V;
    std::array<double,3> A_r = retarded_state.A;

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

    // 3. Halley Correction Step - Lambda to compute error and derivatives from state
    auto compute_halley_residuals = [&](double tau_test, const OrbitalState& state) -> std::tuple<double, double, double, double> {
        double r_vec[3]   = { state.X[0]-X_new[0], state.X[1]-X_new[1], state.X[2]-X_new[2] };
        double R          = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
        if (R < 1e-15) return {NAN, NAN, NAN, NAN};  // Invalid state
        
        double n[3]       = { r_vec[0]/R, r_vec[1]/R, r_vec[2]/R };
        double v_dot_n    = n[0]*state.V[0] + n[1]*state.V[1] + n[2]*state.V[2];
        double F          = (tau_test - t) + R / cs;
        double Fp         = 1.0 + v_dot_n / cs;
        
        double v_sq       = state.V[0]*state.V[0] + state.V[1]*state.V[1] + state.V[2]*state.V[2];
        double n_dot_a    = n[0]*state.A[0] + n[1]*state.A[1] + n[2]*state.A[2];
        double Fpp        = (n_dot_a + (v_sq - v_dot_n*v_dot_n)/R) / cs;
        
        return {F, Fp, Fpp, R};  // Return F, Fp, Fpp, R (used to compute error)
    };

    OrbitalState Halley_state = traj.interpolate(tau);
    auto [F, Fp, Fpp, R] = compute_halley_residuals(tau, Halley_state);
    
    if (!std::isfinite(F)) return false;
    if (std::abs(Fp) < 1e-15) return false;
    
    double denom = 2*Fp*Fp - F*Fpp;
    double tau_old = tau;
    
    if (std::abs(denom) > 1e-20)
        tau -= (2*F*Fp)/denom;    // Halley's method
    else
        tau -= F/Fp;              // fallback to Newton's method

    // 4. Check tau validity and re-interpolate only if tau changed significantly
    if (tau < traj.initial().t || tau > traj.current().t) {
        *tau_out = NAN;
        *err_out = NAN;
        return false;
    }

    // Only re-interpolate if tau changed significantly (avoid redundant 2nd interpolation)
    double dist, err;
    const double tau_tol = 1e-12;  // tolerance to detect if Halley step actually modified tau
    
    if (std::abs(tau - tau_old) > tau_tol) {
        // Halley step moved tau significantly, need fresh interpolation and error
        Halley_state = traj.interpolate(tau);
        auto [F2, Fp2, Fpp2, R2] = compute_halley_residuals(tau, Halley_state);
        
        if (!std::isfinite(F2) || std::abs(Fp2) < 1e-15) return false;
        
        dist = R2;
        err = std::abs((tau - t) + dist / cs);
    } else {
        // Halley step had negligible effect, reuse interpolation result
        dist = R;
        err = std::abs((tau - t) + dist / cs);
    }

    *tau_out = (err > error_tol ? NAN : tau);
    *err_out = err;
    return true;
}





// ===============================================================
// BFS3D loop
// ===============================================================

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

    constexpr int dim   = 3;
    const size_t total_slots = (size_t)Nx * (size_t)Ny * (size_t)Nz * (size_t)MaxRoots;
    std::vector<double> best_err(total_slots, std::numeric_limits<double>::infinity());

    // Small time perturbation used for branching attempts (only in "danger" situations)
    //const double dt_grid = (nt >= 2) ? (t_ptr[1] - t_ptr[0]) : 0.0;
    const double dt_grid = traj.mean_dt();
    const double eps_tr  = (dt_grid != 0.0) ? (0.25 * dt_grid) : 1e-6;

    // Queue holds (i,j,k,slot) where slot is an actual storage slot in Roots(i,j,k,:)
    std::queue<std::tuple<int,int,int,int>> q;
    std::set<std::tuple<int,int,int,int>> visited;
    
    // Track which spatial voxels have been processed to avoid redundant propagation cascades
    std::set<std::tuple<int,int,int>> visited_spatial;

    auto slot_index = [&](int i, int j, int k, int s) -> size_t {
        return idx4d((size_t)i, (size_t)j, (size_t)k, (size_t)s,
                     (size_t)Nx, (size_t)Ny, (size_t)Nz, (size_t)MaxRoots);
    };

    // Voxel root cache: stores sorted roots with lazy rebuild on invalidation
    // This eliminates 222M redundant sorts/allocations in the dedup and branching checks
    struct VoxelRootCache {
        std::vector<double> sorted_roots;
        bool valid = false;
    };
    const size_t total_voxels = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    std::vector<VoxelRootCache> root_cache(total_voxels);

    auto voxel_index = [&](int i, int j, int k) -> size_t {
        return idx3d((size_t)i, (size_t)j, (size_t)k, (size_t)Ny, (size_t)Nz);
    };

    // Invalidate cache for a voxel (called when roots are modified)
    auto invalidate_cache = [&](int i, int j, int k) {
        root_cache[voxel_index(i, j, k)].valid = false;
    };

    // Get sorted roots for a voxel, rebuilding cache if needed
    auto get_sorted_roots = [&](int i, int j, int k) -> const std::vector<double>& {
        size_t v_idx = voxel_index(i, j, k);
        VoxelRootCache& cache = root_cache[v_idx];
        
        if (!cache.valid) {
            cache.sorted_roots.clear();
            cache.sorted_roots.reserve(MaxRoots);
            for (int s = 0; s < MaxRoots; ++s) {
                double tau = Roots(i, j, k, s);
                if (std::isfinite(tau)) {
                    cache.sorted_roots.push_back(tau);
                }
            }
            std::sort(cache.sorted_roots.begin(), cache.sorted_roots.end());
            cache.valid = true;
        }
        
        return cache.sorted_roots;
    };

    // Insert/update a (tau, err) candidate into voxel (i,j,k).
    // Policy:
    //  1) If a new root matches an existing root within unique_tol -> update that slot only if err improves.
    //  2) Otherwise insert into first empty slot if available.
    //  3) Else (voxel full) replace worst slot if err improves.
    // Returns the slot used, or -1 if rejected.
    auto insert_candidate = [&](int i, int j, int k, double tau, double err) -> int {
        if (!std::isfinite(tau) || !std::isfinite(err)) return -1;
        if (err > error_tol) return -1;

        int match_slot = -1;
        int empty_slot = -1;

        // Find match or empty slot
        for (int s = 0; s < MaxRoots; ++s) {
            double ex = Roots(i,j,k,s);
            if (!std::isfinite(ex)) {
                if (empty_slot < 0) empty_slot = s;
                continue;
            }
            if (std::abs(tau - ex) < unique_tol) {
                match_slot = s;
                break;
            }
        }

        // Matched: update only if better
        if (match_slot >= 0) {
            size_t id = slot_index(i,j,k,match_slot);
            if (err < best_err[id]) {
                Roots(i,j,k,match_slot)  = tau;
                Errors(i,j,k,match_slot) = err;
                best_err[id] = err;
                invalidate_cache(i, j, k);
                auto key = std::make_tuple(i,j,k,match_slot);
                if (!visited.count(key)) q.emplace(key);
            }
            return match_slot;
        }

        // Empty: insert
        if (empty_slot >= 0) {
            size_t id = slot_index(i,j,k,empty_slot);
            if (err < best_err[id]) {
                Roots(i,j,k,empty_slot)  = tau;
                Errors(i,j,k,empty_slot) = err;
                best_err[id] = err;
                invalidate_cache(i, j, k);
                auto key = std::make_tuple(i,j,k,empty_slot);
                if (!visited.count(key)) q.emplace(key);
                return empty_slot;
            }
            return -1;
        }

        // Full: replace worst if better
        int worst_slot = 0;
        double worst_e = -1.0;
        for (int s = 0; s < MaxRoots; ++s) {
            double e = best_err[slot_index(i,j,k,s)];
            if (e > worst_e) { worst_e = e; worst_slot = s; }
        }

        if (err < worst_e) {
            size_t id = slot_index(i,j,k,worst_slot);
            Roots(i,j,k,worst_slot)  = tau;
            Errors(i,j,k,worst_slot) = err;
            best_err[id] = err;
            invalidate_cache(i, j, k);
            auto key = std::make_tuple(i,j,k,worst_slot);
            if (!visited.count(key)) q.emplace(key);
            return worst_slot;
        }

        return -1;
    };

    // Seed queue from any existing finite roots; initialize best_err from Errors if present.
    // (Optional: you can deduplicate seeds per voxel here if you want.)
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                for (int s = 0; s < MaxRoots; ++s) {
                    double tau0 = Roots(i,j,k,s);
                    if (!std::isfinite(tau0)) continue;

                    double err0 = Errors(i,j,k,s);
                    if (!std::isfinite(err0)) err0 = 0.0;

                    size_t id = slot_index(i,j,k,s);
                    if (err0 < best_err[id]) {
                        best_err[id] = err0;
                        q.emplace(i,j,k,s);
                    }
                }

    // Timing
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t counter  = 0;
    size_t print_interval = 1000000;

    // "Danger" heuristic parameters (no Fp available yet):
    // If a candidate lands very near an existing root, we suspect fold/cusp proximity and try a second solve.
    const double branch_tol = 5.0 * unique_tol;

    while (!q.empty()) {
        auto [i, j, k, s_here] = q.front();
        q.pop();

        // Skip if already processed
        auto key = std::make_tuple(i, j, k, s_here);
        if (visited.count(key)) continue;
        visited.insert(key);

        double tr_here = Roots(i,j,k,s_here);
        if (!std::isfinite(tr_here)) continue;

        std::array<double,3> X_here = { X[i], Y[j], Z[k] };

        for (int n = 0; n < Neighbours.N; ++n) {
            int di = (int)Neighbours(n,0);
            int dj = (int)Neighbours(n,1);
            int dk = (int)Neighbours(n,2);

            int ni = i + di, nj = j + dj, nk = k + dk;
            if (ni < 0 || nj < 0 || nk < 0 || ni >= Nx || nj >= Ny || nk >= Nz)
                continue;

            std::array<double,3> dX = {X[ni]-X[i], Y[nj]-Y[j], Z[nk]-Z[k]};

            // 1) Primary propagation
            double tauA, errA;
            bool okA = propagate_full(
                tr_here, t, X_here, dX,
                sound_speed, error_tol,
                traj, &tauA, &errA
            );

            bool insertedA = false;
            bool need_branch = false;

            // Stricter deduplication: Skip if candidate root already exists in neighbor
            // Do this EARLY to avoid wasted propagation_full calls
            if (okA && std::isfinite(tauA) && std::isfinite(errA)) {
                const auto& sorted_roots = get_sorted_roots(ni, nj, nk);
                
                if (!sorted_roots.empty()) {
                    bool is_duplicate = false;
                    for (double ex : sorted_roots) {
                        if (std::abs(tauA - ex) < unique_tol) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if (is_duplicate) {
                        counter++;
                        if (counter % print_interval == 0) {
                            auto now       = std::chrono::high_resolution_clock::now();
                            double elapsed = std::chrono::duration<double>(now - start_time).count();
                            double rate    = counter / elapsed;
                            std::cerr << "[bfs3d] " << std::setw(10) << std::setfill('0') << counter << " propagations at rate "
                            << rate/1000000 << " Mzps\n";
                        }
                        continue; // Skip this duplicate root/neighbor pair
                    }
                }
            }

            if (okA && std::isfinite(tauA) && std::isfinite(errA) && errA <= error_tol) {
                // Check proximity to existing roots for branching heuristic
                double min_sep = std::numeric_limits<double>::infinity();
                
                const auto& sorted_roots = get_sorted_roots(ni, nj, nk);
                
                if (!sorted_roots.empty()) {
                    // Binary search: find insertion point and check neighbors
                    auto it = std::lower_bound(sorted_roots.begin(), sorted_roots.end(), tauA);
                    
                    if (it != sorted_roots.end()) {
                        min_sep = std::min(min_sep, std::abs(tauA - *it));
                    }
                    if (it != sorted_roots.begin()) {
                        min_sep = std::min(min_sep, std::abs(tauA - *(it - 1)));
                    }
                }
                
                if (min_sep < branch_tol) need_branch = true;

                int slot_used = insert_candidate(ni, nj, nk, tauA, errA);
                insertedA = (slot_used >= 0);
                
                // If we matched an existing root within unique_tol, branching will just give us 
                // back the same root, so skip it to save computation
                if (insertedA && min_sep < unique_tol) {
                    need_branch = false;
                }
            } else {
                // If primary fails, branching attempt sometimes lands on another sheet
                need_branch = true;
            }

            // 2) Branch attempt only when needed (cheap, minimal-change proxy for cusp handling)
            // We perturb tr slightly; this often causes the predictor/corrector to converge to a different sheet.
            if (need_branch) {
                // Try +eps
                double tauB, errB;
                bool okB = propagate_full(
                    tr_here + eps_tr, t, X_here, dX,
                    sound_speed, error_tol,
                    traj, &tauB, &errB
                    );
                if (okB && std::isfinite(tauB) && std::isfinite(errB) && errB <= error_tol) {
                    insert_candidate(ni, nj, nk, tauB, errB);
                }

                // Try -eps
                double tauC, errC;
                bool okC = propagate_full(
                    tr_here - eps_tr, t, X_here, dX,
                    sound_speed, error_tol,
                    traj, &tauC, &errC
                    );
                if (okC && std::isfinite(tauC) && std::isfinite(errC) && errC <= error_tol) {
                    insert_candidate(ni, nj, nk, tauC, errC);
                }
            }

            counter++;
            if (counter % print_interval == 0) {
                auto now       = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double rate    = counter / elapsed;
                std::cerr << "[bfs3d] " << std::setw(10) << std::setfill('0') << counter << " propagations at rate "
                << rate/1000000 << " Mzps\n";
            }
        }
    }
    //return {Roots, Errors};
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


    const double NonLinearParameter = G * traj.current().M / cs / cs;
    //const double NonLinearParameter = G / cs / cs;

    const int Nx       = Roots.Nx;
    const int Ny       = Roots.Ny;
    const int Nz       = Roots.Nz;
    const int MaxRoots = Roots.Nr;

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

    // Sparse iteration: skip zero/near-zero Alpha voxels to avoid wasted sqrt/division
    // Early threshold check before computing distances = better cache performance
    constexpr double alpha_threshold = 1e-15;
    
    for (std::size_t i = 0; i < Nx; ++i) {
        const double x  = domain.X[i];
        const double rx = x - px;

        for (std::size_t j  = 0; j < Ny; ++j) {
            const double y  = domain.Y[j];
            const double ry = y - py;

            for (std::size_t k  = 0; k < Nz; ++k) {
                const double a  = Alpha(i,j,k);
                
                // Early exit: skip zero/near-zero Alpha before computing distances
                // This is crucial for sparse domains where most voxels have Alpha ≈ 0
                if (a <= alpha_threshold) continue;

                const double z  = domain.Z[k];
                const double rz = z - pz;
                
                const double r2 = rx*rx + ry*ry + rz*rz;
                const double r  = std::sqrt(r2);

                if (r <= rmin) continue;

                const double r3 = r2 * r;
                const double coeff = (G * M * rho_0 * a * dV) / r3;

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
    }
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

    