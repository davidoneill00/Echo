// ===============================================================
// Method.cpp  —  C++ implementation of BFS3D algorithm
// using pybind11 for Python integrability
// ===============================================================

#include <pybind11/pybind11.h> // to bind c++ with python
#include <pybind11/numpy.h>    // to bind c++ with numpy
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>

namespace py = pybind11;       // rename for convenience


// Helper interpolation for uniform time grids
inline void interp_linear_uniform(
    double xp0,            // initial value in array
    double dx,             // array spacing
    const double* fp,      // constant pointer to address of the first element of fp
    int n,                 // number of data points
    int dim,               // number of dimensions in data set (although we only interp in 1d)
    double x,              // quiery position
    double* out)           // pointer to output value
{
    // Compute linear interpolation index
    double xi         = (x - xp0) / dx;
    int i             = static_cast<int>(xi);   // truncate decimals to int and save as i
    if (i < 0) i      = 0;
    if (i >= n - 1) i = n - 2;
    double w = xi - i;

    // address of ith element of fp is fp + i * dim
    const double* f0 = fp + i * dim;
    const double* f1 = fp + (i + 1) * dim;
    for (int d = 0; d < dim; ++d) // loop
        out[d] = f0[d] + w * (f1[d] - f0[d]); // then we access the address given as double* out
}


// Helper function for tracking visited cells in BFS algorithm
inline size_t idx3d(size_t i, size_t j, size_t k,
                    size_t Nx, size_t Ny, size_t Nz) {
    return (i * Ny + j) * Nz + k;
}
inline size_t idx4d(size_t i, size_t j, size_t k, size_t r,
                    size_t Nx, size_t Ny, size_t Nz, size_t Nr) {
    return ((i * Ny + j) * Nz + k) * Nr + r;
}

// ===============================================================
// Propagate function 
// ===============================================================
inline bool propagate_full_cpp(
    double tr, 
    double t,
    const double* X, 
    const double* dX,
    double cs,
    double error_tol,
    //double care_control,
    const double* t_arr, 
    int nt,
    const double* x_arr,
    const double* v_arr,
    const double* a_arr,
    int dim,
    double* tau_out, 
    double* err_out)
{

    // 0. Initialise
    double X_r[3];
    double V_r[3];
    double A_r[3];
    double X_rel[3];
    double Projection[3];
    double PV=0;
    double PA=0;
    double Px=0;
    double Ax=0;
    double Vx=0;
    double xx=0;
    double VV=0;

    double xp0 = t_arr[0];
    double dx  = t_arr[1] - t_arr[0];
    if (!std::isfinite(tr)) return false;

    // 2. Predictor Step
    interp_linear_uniform(xp0, dx, x_arr, nt, dim, tr, X_r); // saves position 
    interp_linear_uniform(xp0, dx, v_arr, nt, dim, tr, V_r); // saves velocity 
    interp_linear_uniform(xp0, dx, a_arr, nt, dim, tr, A_r); // saves acceleration 
    for (int d = 0; d < dim; ++d) X_rel[d] = X_r[d] - X[d];
    double X_rel_norm = std::sqrt(X_rel[0]*X_rel[0] + X_rel[1]*X_rel[1] + X_rel[2]*X_rel[2]);
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
    double X_tau[3], V_tau[3], A_tau[3];
    interp_linear_uniform(xp0, dx, x_arr, nt, dim, tau, X_tau);
    interp_linear_uniform(xp0, dx, v_arr, nt, dim, tau, V_tau);
    interp_linear_uniform(xp0, dx, a_arr, nt, dim, tau, A_tau);

    double r_vec[3]   = { X_tau[0]-X_new[0], X_tau[1]-X_new[1], X_tau[2]-X_new[2] };
    double R          = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    double dist       = R;
    double first_err  = std::abs((tau - t) + dist / cs);
    double n[3]       = { r_vec[0]/R, r_vec[1]/R, r_vec[2]/R };
    double v_dot_n    = n[0]*V_tau[0] + n[1]*V_tau[1] + n[2]*V_tau[2];
    double F          = (tau - t) + R / cs;
    double Fp         = 1.0 + v_dot_n / cs;
    double v_sq       = V_tau[0]*V_tau[0] + V_tau[1]*V_tau[1] + V_tau[2]*V_tau[2];
    double n_dot_a    = n[0]*A_tau[0] + n[1]*A_tau[1] + n[2]*A_tau[2];
    double Fpp        = (n_dot_a + (v_sq - v_dot_n*v_dot_n)/R) / cs;
    double denom      = 2*Fp*Fp - F*Fpp;

    if (std::abs(denom) > 1e-20)
        tau -= (2*F*Fp)/denom;    // Halley's method
    else
        tau -= F/Fp;              // fallback to Newton's method

    // 3. Define error with refined tau
    interp_linear_uniform(xp0, dx, x_arr, nt, dim, tau, X_tau); // Overwrites X_tau
    interp_linear_uniform(xp0, dx, v_arr, nt, dim, tau, V_tau);
    
    r_vec[0]   = X_tau[0]-X_new[0];
    r_vec[1]   = X_tau[1]-X_new[1];
    r_vec[2]   = X_tau[2]-X_new[2];
    //r_vec[3]    = { X_tau[0]-X_new[0], X_tau[1]-X_new[1], X_tau[2]-X_new[2] };
    R           = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    dist        = R;
    n[0]        = r_vec[0]/R;
    n[1]        = r_vec[1]/R;
    n[2]        = r_vec[2]/R;
    v_dot_n    = n[0]*V_tau[0] + n[1]*V_tau[1] + n[2]*V_tau[2];
    F          = (tau - t) + R / cs;
    Fp         = 1.0 + v_dot_n / cs;
    v_sq       = V_tau[0]*V_tau[0] + V_tau[1]*V_tau[1] + V_tau[2]*V_tau[2];
    n_dot_a    = n[0]*A_tau[0] + n[1]*A_tau[1] + n[2]*A_tau[2];
    Fpp        = (n_dot_a + (v_sq - v_dot_n*v_dot_n)/R) / cs;

    if (tau < t_arr[0] || tau > t_arr[nt-1] || tau > t) {
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

    //std::cout << "error1 " << first_err << " error2 " << second_err << "\n";
    *tau_out = (err > error_tol ? NAN : tau);
    *err_out = err;
    return true;
}


// ===============================================================
// BFS3D loop
// ===============================================================

py::tuple bfs3d( 
    py::array_t<double> Roots,       // arrays from python
    py::array_t<double> Errors,
    py::array_t<double> X,
    py::array_t<double> Y,
    py::array_t<double> Z,
    py::array_t<double> Neighbors,
    double t,
    double sound_speed,
    double error_tol,
    double unique_tol,
    py::array_t<double> t_arr,
    py::array_t<double> x_arr,
    py::array_t<double> v_arr,
    py::array_t<double> a_arr)
{
    auto buf_roots     = Roots.mutable_unchecked<4>();
    auto buf_error     = Errors.mutable_unchecked<4>();
    auto buf_neighbors = Neighbors.unchecked<2>();
    auto buf_X         = X.unchecked<1>();
    auto buf_Y         = Y.unchecked<1>();
    auto buf_Z         = Z.unchecked<1>();

    const int Nx       = (int)buf_roots.shape(0);
    const int Ny       = (int)buf_roots.shape(1);
    const int Nz       = (int)buf_roots.shape(2);
    const int MaxRoots = (int)buf_roots.shape(3);

    const size_t total_slots = (size_t)Nx * (size_t)Ny * (size_t)Nz * (size_t)MaxRoots;

    // Instead of labelling each cell as visited, keep best error achieved for each (i,j,k,slot)
    std::vector<double> best_err(total_slots, std::numeric_limits<double>::infinity());

    // Raw pointers for propagation
    const double* t_ptr = t_arr.data();
    const double* x_ptr = x_arr.data();
    const double* v_ptr = v_arr.data();
    const double* a_ptr = a_arr.data();
    const int nt  = (int)t_arr.shape(0);
    const int dim = 3;

    // Small time perturbation used for branching attempts (only in "danger" situations)
    const double dt_grid = (nt >= 2) ? (t_ptr[1] - t_ptr[0]) : 0.0;
    const double eps_tr  = (dt_grid != 0.0) ? (0.25 * dt_grid) : 1e-6;

    // Queue holds (i,j,k,slot) where slot is an actual storage slot in Roots(i,j,k,:)
    std::queue<std::tuple<int,int,int,int>> q;

    auto slot_index = [&](int i, int j, int k, int s) -> size_t {
        return idx4d((size_t)i, (size_t)j, (size_t)k, (size_t)s,
                     (size_t)Nx, (size_t)Ny, (size_t)Nz, (size_t)MaxRoots);
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
            double ex = buf_roots(i,j,k,s);
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
                buf_roots(i,j,k,match_slot) = tau;
                buf_error(i,j,k,match_slot) = err;
                best_err[id] = err;
                q.emplace(i,j,k,match_slot);
            }
            return match_slot;
        }

        // Empty: insert
        if (empty_slot >= 0) {
            size_t id = slot_index(i,j,k,empty_slot);
            if (err < best_err[id]) {
                buf_roots(i,j,k,empty_slot) = tau;
                buf_error(i,j,k,empty_slot) = err;
                best_err[id] = err;
                q.emplace(i,j,k,empty_slot);
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
            buf_roots(i,j,k,worst_slot) = tau;
            buf_error(i,j,k,worst_slot) = err;
            best_err[id] = err;
            q.emplace(i,j,k,worst_slot);
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
                    double tau0 = buf_roots(i,j,k,s);
                    if (!std::isfinite(tau0)) continue;

                    double err0 = buf_error(i,j,k,s);
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

        double tr_here = buf_roots(i,j,k,s_here);
        if (!std::isfinite(tr_here)) continue;

        double X_here[3] = { buf_X(i), buf_Y(j), buf_Z(k) };

        for (int n = 0; n < (int)buf_neighbors.shape(0); ++n) {
            int di = (int)buf_neighbors(n,0);
            int dj = (int)buf_neighbors(n,1);
            int dk = (int)buf_neighbors(n,2);

            int ni = i + di, nj = j + dj, nk = k + dk;
            if (ni < 0 || nj < 0 || nk < 0 || ni >= Nx || nj >= Ny || nk >= Nz)
                continue;

            double dX[3] = {
                buf_X(ni) - buf_X(i),
                buf_Y(nj) - buf_Y(j),
                buf_Z(nk) - buf_Z(k)
            };

            // 1) Primary propagation
            double tauA, errA;
            bool okA = propagate_full_cpp(
                tr_here, t, X_here, dX,
                sound_speed, error_tol,
                t_ptr, nt, x_ptr, v_ptr, a_ptr, dim,
                &tauA, &errA
            );

            bool insertedA = false;
            bool need_branch = false;

            if (okA && std::isfinite(tauA) && std::isfinite(errA) && errA <= error_tol) {
                // Check proximity to existing roots in target voxel to decide if we need branch attempt
                double min_sep = std::numeric_limits<double>::infinity();
                for (int s = 0; s < MaxRoots; ++s) {
                    double ex = buf_roots(ni,nj,nk,s);
                    if (!std::isfinite(ex)) continue;
                    double sep = std::abs(tauA - ex);
                    if (sep < min_sep) min_sep = sep;
                }
                if (min_sep < branch_tol) need_branch = true;

                int slot_used = insert_candidate(ni, nj, nk, tauA, errA);
                insertedA = (slot_used >= 0);
            } else {
                // If primary fails, branching attempt sometimes lands on another sheet
                need_branch = true;
            }

            // 2) Branch attempt only when needed (cheap, minimal-change proxy for cusp handling)
            // We perturb tr slightly; this often causes the predictor/corrector to converge to a different sheet.
            if (need_branch) {
                // Try +eps
                double tauB, errB;
                bool okB = propagate_full_cpp(
                    tr_here + eps_tr, t, X_here, dX,
                    sound_speed, error_tol,
                    t_ptr, nt, x_ptr, v_ptr, a_ptr, dim,
                    &tauB, &errB
                );
                if (okB && std::isfinite(tauB) && std::isfinite(errB) && errB <= error_tol) {
                    insert_candidate(ni, nj, nk, tauB, errB);
                }

                // Try -eps
                double tauC, errC;
                bool okC = propagate_full_cpp(
                    tr_here - eps_tr, t, X_here, dX,
                    sound_speed, error_tol,
                    t_ptr, nt, x_ptr, v_ptr, a_ptr, dim,
                    &tauC, &errC
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

    return py::make_tuple(Roots, Errors);
}



// Translate roots to density perturbations Alpha
py::tuple Alpha3d(
    py::array_t<double> Roots,      
    py::array_t<double> X,
    py::array_t<double> Y,
    py::array_t<double> Z,
    //py::array_t<double> OriginPosition,
    py::array_t<double> t_arr,
    py::array_t<double> x_arr,
    py::array_t<double> v_arr,
    const double cs)
{
    auto buf_roots     = Roots.unchecked<4>(); // fast access to Roots without safety checks
    auto buf_X         = X.unchecked<1>();
    auto buf_Y         = Y.unchecked<1>();
    auto buf_Z         = Z.unchecked<1>();
    auto buf_t         = t_arr.unchecked<1>();  // if 1D array

    double xp0          = buf_t(0);
    double dx           = buf_t(1) - buf_t(0);
    int nt              = t_arr.shape(0);
    int dim             = 3;
    const double* x_ptr = x_arr.data();
    const double* v_ptr = v_arr.data();


    size_t Nx           = buf_roots.shape(0);
    size_t Ny           = buf_roots.shape(1);
    size_t Nz           = buf_roots.shape(2);
    size_t MaxRoots     = buf_roots.shape(3);


    py::array_t<double> Alpha({Nx, Ny, Nz});
    auto buf_alpha = Alpha.mutable_unchecked<3>();

    py::array_t<int> NRoots(std::vector<size_t>{Nx, Ny, Nz});
    auto buf_nrt   = NRoots.mutable_unchecked<3>();


    const double tol = 5e-7;

for (int i = 0; i < Nx; ++i)
    for (int j = 0; j < Ny; ++j)
        for (int k = 0; k < Nz; ++k) {

            double rootweight = 0.0;
            int rootnumber = 0;

            // Temporary storage of unique roots for this voxel
            std::vector<double> unique_roots;
            unique_roots.reserve(MaxRoots);

            for (int r = 0; r < MaxRoots; ++r) {
                double tr = buf_roots(i, j, k, r);
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
                double x = buf_X(i);
                double y = buf_Y(j);
                double z = buf_Z(k);

                double X_r[3], V_r[3];
                interp_linear_uniform(xp0, dx, x_ptr, nt, dim, tr, X_r);
                interp_linear_uniform(xp0, dx, v_ptr, nt, dim, tr, V_r);

                double X_rel[3] = {X_r[0] - x, X_r[1] - y, X_r[2] - z};
                double X_rel_norm = std::sqrt(
                    X_rel[0]*X_rel[0] + X_rel[1]*X_rel[1] + X_rel[2]*X_rel[2]);

                double denom = std::abs(X_rel_norm +
                    (X_rel[0]*V_r[0] + X_rel[1]*V_r[1] + X_rel[2]*V_r[2]) / cs);

                rootweight += 1.0 / denom;
                rootnumber += 1;
            }

            buf_alpha(i,j,k) = rootweight;
            buf_nrt(i,j,k) = rootnumber;
        }

    return py::make_tuple(Alpha, NRoots);
}



py::array_t<double> UniqueArray_4d(
    py::array_t<double> A,      
    const double unique_tol)
{
    // Access array safely (read-only)
    auto buf_A = A.unchecked<4>();
    size_t Nx = buf_A.shape(0);
    size_t Ny = buf_A.shape(1);
    size_t Nz = buf_A.shape(2);
    size_t Nr = buf_A.shape(3);

    // Create output array with same shape, filled with NaN
    auto info = A.request();
    py::array_t<double> UniqueA(info.shape);
    std::fill_n(UniqueA.mutable_data(), UniqueA.size(), NAN);
    auto buf_unique = UniqueA.mutable_unchecked<4>();

    // Loop through spatial cells
    for (size_t i = 0; i < Nx; ++i)
        for (size_t j = 0; j < Ny; ++j)
            for (size_t k = 0; k < Nz; ++k) {

                // Compare roots within this voxel
                for (size_t ri = 0; ri < Nr; ++ri) {
                    double first = buf_A(i, j, k, ri);
                    if (!std::isfinite(first))
                        continue;

                    bool unique = true;
                    for (size_t rj = ri + 1; rj < Nr; ++rj) {
                        double second = buf_A(i, j, k, rj);
                        if (std::isfinite(second) &&
                            std::abs(first - second) < unique_tol) {
                            unique = false;
                            break;
                        }
                    }

                    if (unique)
                        buf_unique(i, j, k, ri) = first;
                }
            }

    return UniqueA;
}




py::tuple propagate_full(
    double tr,
    double t,
    py::array_t<double> X,
    py::array_t<double> dX,
    double cs,
    double error_tol,
    py::array_t<double> t_arr,
    py::array_t<double> x_arr,
    py::array_t<double> v_arr,
    py::array_t<double> a_arr)
{
    auto buf_X  = X.unchecked<1>();
    auto buf_dX = dX.unchecked<1>();

    const double* t_ptr = t_arr.data();
    const double* x_ptr = x_arr.data();
    const double* v_ptr = v_arr.data();
    const double* a_ptr = a_arr.data();
    int nt  = t_arr.shape(0);
    int dim = 3;

    double tau = NAN, err = NAN;
    bool ok = propagate_full_cpp(
        tr, t,
        buf_X.data(0),
        buf_dX.data(0),
        cs,
        error_tol,
        t_ptr, nt,
        x_ptr, v_ptr, a_ptr, dim,
        &tau, &err
    );

    return py::make_tuple(ok, tau, err);
}



// ===============================================================
// Bind the function to Python
// ===============================================================
PYBIND11_MODULE(Method_cpp, m) {
    m.doc() = "C++ BFS3D solver for RoutineSupreme.py";
    m.def("bfs3d",          &bfs3d, "Perform BFS root propagation");
    m.def("propagate_full", &propagate_full, "Propagate single BFS step");
    m.def("Alpha3d",        &Alpha3d, "Translate roots to density perturbations Alpha");
    m.def("UniqueArray_4d", &UniqueArray_4d, "Remove near-duplicate entries along last axis");

}

