#pragma once

#include <vector>
#include <limits>
#include <cstddef>
#include <algorithm> // lower_bound

inline constexpr double G = 1.0; // Newton G

// inline std::size_t left_index_bracketed(const std::vector<double>& ts, double t) {
//     if (ts.size() < 2) return 0;          // or throw
//     if (t <= ts.front()) return 0;
//     if (t >= ts.back())  return ts.size() - 2;

//     auto it = std::lower_bound(ts.begin(), ts.end(), t);
//     return static_cast<std::size_t>(it - ts.begin()) - 1;
// }

inline std::vector<double> linspace(double start, double end, std::size_t num) {
    std::vector<double> result(num);
    if (num == 0) return result;
    if (num == 1) { result[0] = start; return result; }

    double step = (end - start) / (num - 1);
    for (std::size_t i = 0; i < num; ++i) result[i] = start + i * step;
    return result;
}

inline void interp_linear_uniform(double xp0, double dx, const double* fp,
                                  int n, int dim, double x, double* out) {
    double xi = (x - xp0) / dx;
    int i = static_cast<int>(xi);
    if (i < 0) i = 0;
    if (i >= n - 1) i = n - 2;
    double w = xi - i;

    const double* f0 = fp + i * dim;
    const double* f1 = fp + (i + 1) * dim;
    for (int d = 0; d < dim; ++d)
        out[d] = f0[d] + w * (f1[d] - f0[d]);
}

inline std::size_t idx3d(std::size_t i, std::size_t j, std::size_t k,
                         std::size_t Ny, std::size_t Nz) {
    return (i * Ny + j) * Nz + k;
}

inline std::size_t idx4d(std::size_t i, std::size_t j, std::size_t k, std::size_t r,
                         std::size_t Nx, std::size_t Ny, std::size_t Nz, std::size_t Nr) {
    return ((i * Ny + j) * Nz + k) * Nr + r;
}

struct Mat4D {
    int Nx, Ny, Nz, Nr;
    std::vector<double> a;

    Mat4D(int nx, int ny, int nz, int nr,
          double init = std::numeric_limits<double>::quiet_NaN())
        : Nx(nx), Ny(ny), Nz(nz), Nr(nr), a((std::size_t)nx*ny*nz*nr, init) {}

    double& operator()(int i,int j,int k,int r) {
        return a[(((std::size_t)i*Ny + j)*Nz + k)*Nr + r];
    }
    double operator()(int i,int j,int k,int r) const {
        return a[(((std::size_t)i*Ny + j)*Nz + k)*Nr + r];
    }
};

struct Mat3D {
    int Nx, Ny, Nz;
    std::vector<double> a;

    Mat3D(int nx, int ny, int nz, double init = 0.0)
        : Nx(nx), Ny(ny), Nz(nz), a((std::size_t)nx*ny*nz, init) {}

    double& operator()(int i,int j,int k) {
        return a[((std::size_t)i*Ny + j)*Nz + k];
    }
    double operator()(int i,int j,int k) const {
        return a[((std::size_t)i*Ny + j)*Nz + k];
    }
};

struct Mat2I {
    int N, M;
    std::vector<int> a;

    Mat2I(int n, int m) : N(n), M(m), a((std::size_t)n*m) {}

    int& operator()(int i,int j){ return a[(std::size_t)i*M + j]; }
    int  operator()(int i,int j) const { return a[(std::size_t)i*M + j]; }
};


inline std::vector<double> flatten_alpha(const Mat3D& A)
{
    const std::size_t Nx = A.Nx;
    const std::size_t Ny = A.Ny;
    const std::size_t Nz = A.Nz;

    std::vector<double> flat;
    flat.reserve(Nx * Ny * Nz);

    for (std::size_t i = 0; i < Nx; ++i)
        for (std::size_t j = 0; j < Ny; ++j)
            for (std::size_t k = 0; k < Nz; ++k)
                flat.push_back(A(i,j,k));

    return flat;
}