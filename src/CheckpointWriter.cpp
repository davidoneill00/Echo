#include "CheckpointWriter.hpp"
#include "Checkpoint.hpp"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

// ----------------------------
// Minimal RAII for HDF5 handles (CPP-only)
// ----------------------------
struct H5FileHandle {
    hid_t id = -1;

    H5FileHandle() = default;
    explicit H5FileHandle(hid_t x) : id(x) {}

    H5FileHandle(const H5FileHandle&) = delete;
    H5FileHandle& operator=(const H5FileHandle&) = delete;

    H5FileHandle(H5FileHandle&& o) noexcept : id(std::exchange(o.id, -1)) {}
    H5FileHandle& operator=(H5FileHandle&& o) noexcept {
        if (this != &o) {
            close();
            id = std::exchange(o.id, -1);
        }
        return *this;
    }

    ~H5FileHandle() { close(); }

    void close() {
        if (id >= 0) {
            H5Fclose(id);
            id = -1;
        }
    }
};

struct H5ObjHandle {
    hid_t id = -1;
    herr_t (*closer)(hid_t) = nullptr;

    H5ObjHandle() = default;
    H5ObjHandle(hid_t x, herr_t(*c)(hid_t)) : id(x), closer(c) {}

    H5ObjHandle(const H5ObjHandle&) = delete;
    H5ObjHandle& operator=(const H5ObjHandle&) = delete;

    H5ObjHandle(H5ObjHandle&& o) noexcept
        : id(std::exchange(o.id, -1)), closer(o.closer) {
        o.closer = nullptr;
    }

    H5ObjHandle& operator=(H5ObjHandle&& o) noexcept {
        if (this != &o) {
            close();
            id = std::exchange(o.id, -1);
            closer = o.closer;
            o.closer = nullptr;
        }
        return *this;
    }

    ~H5ObjHandle() { close(); }

    void close() {
        if (id >= 0 && closer) {
            closer(id);
            id = -1;
            closer = nullptr;
        }
    }
};

// ----------------------------
// Helpers
// ----------------------------
static void throw_h5(const std::string& msg) {
    throw std::runtime_error("HDF5 error: " + msg);
}

static bool link_exists(hid_t loc, const std::string& path) {
    htri_t ex = H5Lexists(loc, path.c_str(), H5P_DEFAULT);
    if (ex < 0) throw_h5("H5Lexists failed for " + path);
    return ex > 0;
}

static void ensure_groups(hid_t file, const std::string& full_path) {
    if (full_path.empty() || full_path[0] != '/') return;

    std::string cur;
    cur.reserve(full_path.size());

    std::size_t i = 1;
    while (i < full_path.size()) {
        std::size_t j = full_path.find('/', i);
        if (j == std::string::npos) j = full_path.size();
        std::string part = full_path.substr(i, j - i);
        i = j + 1;

        if (part.empty()) continue;

        cur += "/";
        cur += part;

        if (!link_exists(file, cur)) {
            hid_t gid = H5Gcreate2(file, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (gid < 0) throw_h5("H5Gcreate2 failed for " + cur);
            H5Gclose(gid);
        }
    }
}

static void ensure_parent_groups(hid_t file, const std::string& dataset_path) {
    auto pos = dataset_path.find_last_of('/');
    if (pos == std::string::npos || pos == 0) return;
    ensure_groups(file, dataset_path.substr(0, pos));
}

static void delete_if_exists(hid_t file, const std::string& path) {
    if (link_exists(file, path)) {
        if (H5Ldelete(file, path.c_str(), H5P_DEFAULT) < 0)
            throw_h5("H5Ldelete failed for " + path);
    }
}

// ----------------------------
// Dataset properties helper with chunking and compression
// ----------------------------
static hid_t create_dataset_properties(const hsize_t* dims, int ndim) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl < 0) return -1;

    // Set appropriate chunk sizes based on dimensionality
    hsize_t chunk_dims[H5S_MAX_RANK];
    if (ndim == 3) {
        // For 3D (Alpha data): 64x64x64 chunks (~32 MB each)
        chunk_dims[0] = std::min(dims[0], (hsize_t)64);
        chunk_dims[1] = std::min(dims[1], (hsize_t)64);
        chunk_dims[2] = std::min(dims[2], (hsize_t)64);
    } else if (ndim == 2) {
        // For 2D (trajectory): 256x8 chunks
        chunk_dims[0] = std::min(dims[0], (hsize_t)256);
        chunk_dims[1] = std::min(dims[1], (hsize_t)8);
    } else if (ndim == 1) {
        // For 1D (vectors): 1024-element chunks
        chunk_dims[0] = std::min(dims[0], (hsize_t)1024);
    }

    // Set chunking
    if (H5Pset_chunk(dcpl, ndim, chunk_dims) < 0) {
        H5Pclose(dcpl);
        return -1;
    }

    // Enable GZIP compression level 4 (good compression, still fast)
    if (H5Pset_deflate(dcpl, 4) < 0) {
        H5Pclose(dcpl);
        return -1;
    }

    return dcpl;
}

// ----------------------------
// Writers
// ----------------------------
static void write_scalar_double(hid_t file, const std::string& path, double value) {
    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[1] = {1};
    H5ObjHandle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple scalar failed");

    hid_t dcpl = create_dataset_properties(dims, 1);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_DOUBLE, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    if (H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void write_scalar_int(hid_t file, const std::string& path, int value) {
    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[1] = {1};
    H5ObjHandle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple scalar failed");

    hid_t dcpl = create_dataset_properties(dims, 1);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_INT, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    if (H5Dwrite(dset.id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void write_scalar_u64(hid_t file, const std::string& path, std::uint64_t value) {
    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[1] = {1};
    H5ObjHandle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple scalar failed");

    hid_t dcpl = create_dataset_properties(dims, 1);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_UINT64, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    if (H5Dwrite(dset.id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void write_vec1d_double(hid_t file, const std::string& path, const std::vector<double>& v) {
    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[1] = { static_cast<hsize_t>(v.size()) };
    H5ObjHandle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple 1d failed");

    hid_t dcpl = create_dataset_properties(dims, 1);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_DOUBLE, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    const void* buf = v.empty() ? nullptr : static_cast<const void*>(v.data());
    if (H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void write_flat_2d_double(hid_t file, const std::string& path,
                                 const std::vector<double>& flat,
                                 std::size_t dim0, std::size_t dim1) {
    if (flat.size() != dim0 * dim1)
        throw std::runtime_error("write_flat_2d_double: flat size mismatch for " + path);

    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[2] = { static_cast<hsize_t>(dim0), static_cast<hsize_t>(dim1) };
    H5ObjHandle space(H5Screate_simple(2, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple 2d failed");

    hid_t dcpl = create_dataset_properties(dims, 2);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_DOUBLE, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    const void* buf = flat.empty() ? nullptr : static_cast<const void*>(flat.data());
    if (H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void write_flat_3d_double(hid_t file, const std::string& path,
                                 const std::vector<double>& flat,
                                 std::size_t nx, std::size_t ny, std::size_t nz) {
    if (flat.size() != nx * ny * nz)
        throw std::runtime_error("write_flat_3d_double: flat size mismatch for " + path);

    ensure_parent_groups(file, path);
    delete_if_exists(file, path);

    hsize_t dims[3] = { static_cast<hsize_t>(nx), static_cast<hsize_t>(ny), static_cast<hsize_t>(nz) };
    H5ObjHandle space(H5Screate_simple(3, dims, nullptr), H5Sclose);
    if (space.id < 0) throw_h5("H5Screate_simple 3d failed");

    hid_t dcpl = create_dataset_properties(dims, 3);
    if (dcpl < 0) throw_h5("create_dataset_properties failed for " + path);

    H5ObjHandle dset(
        H5Dcreate2(file, path.c_str(), H5T_NATIVE_DOUBLE, space.id,
                   H5P_DEFAULT, dcpl, H5P_DEFAULT),
        H5Dclose
    );
    H5Pclose(dcpl);

    if (dset.id < 0) throw_h5("H5Dcreate2 failed for " + path);

    const void* buf = flat.empty() ? nullptr : static_cast<const void*>(flat.data());
    if (H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0)
        throw_h5("H5Dwrite failed for " + path);
}

static void ensure_base_groups(hid_t file) {
    ensure_groups(file, "/params/fixed");
    ensure_groups(file, "/params/runtime_saved");
    ensure_groups(file, "/domain");
    ensure_groups(file, "/trajectory");
    ensure_groups(file, "/force");
    ensure_groups(file, "/alpha");
}

// ----------------------------
// save_checkpoint implementation
// ----------------------------
void save_checkpoint(
    const std::string& out_dir,
    int checkpoint_index,
    const FixedParams& fixed,
    const RuntimeParams& runtime,
    const std::vector<double>& t_series,
    const std::vector<double>& M_series,
    const std::vector<std::array<double,3>>& X_series,
    const std::vector<std::array<double,3>>& V_series,
    const std::vector<std::array<double,3>>& A_series,
    int iteration_saved,
    int RecordTrajectoryCadence_used,
    const std::vector<double>& force_t,
    const std::vector<std::array<double,3>>& force_F,
    const std::vector<AlphaSnapshot>& alpha_snapshots
) {
    const std::size_t n = t_series.size();
    if (n == 0) throw std::runtime_error("save_checkpoint: empty trajectory");
    if (M_series.size() != n || X_series.size() != n || V_series.size() != n || A_series.size() != n)
        throw std::runtime_error("save_checkpoint: trajectory size mismatch");
    if (!force_t.empty() && force_F.size() != force_t.size())
        throw std::runtime_error("save_checkpoint: force size mismatch");

    std::filesystem::create_directories(out_dir);

    const std::string tmp_path = checkpoint_tmp_filename(out_dir, checkpoint_index);
    const std::string fin_path = checkpoint_filename(out_dir, checkpoint_index);

    H5FileHandle f(H5Fcreate(tmp_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    if (f.id < 0) throw_h5("H5Fcreate failed for " + tmp_path);

    ensure_base_groups(f.id);

    // /params/fixed
    write_scalar_double(f.id, "/params/fixed/seed_fraction", fixed.seed_fraction);
    write_scalar_int   (f.id, "/params/fixed/max_number_of_roots", fixed.max_number_of_roots);

    write_scalar_double(f.id, "/params/fixed/initial_state/t", fixed.initial_state.t);
    write_scalar_double(f.id, "/params/fixed/initial_state/M", fixed.initial_state.M);
    write_vec1d_double(f.id, "/params/fixed/initial_state/X",
                       {fixed.initial_state.X[0], fixed.initial_state.X[1], fixed.initial_state.X[2]});
    write_vec1d_double(f.id, "/params/fixed/initial_state/V",
                       {fixed.initial_state.V[0], fixed.initial_state.V[1], fixed.initial_state.V[2]});
    write_vec1d_double(f.id, "/params/fixed/initial_state/A",
                       {fixed.initial_state.A[0], fixed.initial_state.A[1], fixed.initial_state.A[2]});

    write_scalar_double(f.id, "/params/fixed/sound_speed", fixed.sound_speed);
    write_scalar_double(f.id, "/params/fixed/rho0", fixed.rho0);
    write_scalar_double(f.id, "/params/fixed/rmin", fixed.rmin);
    write_scalar_double(f.id, "/params/fixed/error_tol", fixed.error_tol);
    write_scalar_double(f.id, "/params/fixed/unique_tol", fixed.unique_tol);

    write_scalar_u64(f.id, "/params/fixed/domain/Nx", static_cast<std::uint64_t>(fixed.Nx));
    write_scalar_u64(f.id, "/params/fixed/domain/Ny", static_cast<std::uint64_t>(fixed.Ny));
    write_scalar_u64(f.id, "/params/fixed/domain/Nz", static_cast<std::uint64_t>(fixed.Nz));
    write_vec1d_double(f.id, "/params/fixed/domain/range_x", {fixed.range_x[0], fixed.range_x[1]});
    write_vec1d_double(f.id, "/params/fixed/domain/range_y", {fixed.range_y[0], fixed.range_y[1]});
    write_vec1d_double(f.id, "/params/fixed/domain/range_z", {fixed.range_z[0], fixed.range_z[1]});

    // /params/runtime_saved
    write_scalar_double(f.id, "/params/runtime_saved/timelimiter", runtime.timelimiter);
    write_scalar_int   (f.id, "/params/runtime_saved/finite_timestep", runtime.finite_timestep ? 1 : 0);
    write_scalar_int   (f.id, "/params/runtime_saved/RecordTrajectoryCadence", runtime.RecordTrajectoryCadence);

    // /trajectory
    write_vec1d_double(f.id, "/trajectory/t", t_series);
    write_vec1d_double(f.id, "/trajectory/M", M_series);

    const auto X_flat = flatten_vec3(X_series);
    const auto V_flat = flatten_vec3(V_series);
    const auto A_flat = flatten_vec3(A_series);

    write_flat_2d_double(f.id, "/trajectory/X", X_flat, n, 3);
    write_flat_2d_double(f.id, "/trajectory/V", V_flat, n, 3);
    write_flat_2d_double(f.id, "/trajectory/A", A_flat, n, 3);

    write_scalar_int(f.id, "/trajectory/iteration", iteration_saved);
    write_scalar_int(f.id, "/trajectory/RecordTrajectoryCadence_used", RecordTrajectoryCadence_used);

    // /force
    if (!force_t.empty()) {
        write_vec1d_double(f.id, "/force/t", force_t);
        const auto F_flat = flatten_vec3(force_F);
        write_flat_2d_double(f.id, "/force/F", F_flat, force_t.size(), 3);
    } else {
        write_vec1d_double(f.id, "/force/t", {});
        write_flat_2d_double(f.id, "/force/F", {}, 0, 3);
    }

    // /alpha — one group per refinement level
    write_scalar_int(f.id, "/alpha/num_levels", static_cast<int>(alpha_snapshots.size()));
    for (const auto& snap : alpha_snapshots) {
        if (snap.alpha_flat.size() != snap.Nx * snap.Ny * snap.Nz)
            throw std::runtime_error("save_checkpoint: alpha size mismatch at level "
                                     + std::to_string(snap.level));
        const std::string base = "/alpha/level_" + std::to_string(snap.level);
        write_flat_3d_double(f.id, base + "/data",
                             snap.alpha_flat, snap.Nx, snap.Ny, snap.Nz);
        write_scalar_double(f.id, base + "/meta/t",  snap.t_alpha);
        write_scalar_u64   (f.id, base + "/meta/Nx", static_cast<std::uint64_t>(snap.Nx));
        write_scalar_u64   (f.id, base + "/meta/Ny", static_cast<std::uint64_t>(snap.Ny));
        write_scalar_u64   (f.id, base + "/meta/Nz", static_cast<std::uint64_t>(snap.Nz));
        write_vec1d_double (f.id, base + "/meta/range_x", {snap.range_x[0], snap.range_x[1]});
        write_vec1d_double (f.id, base + "/meta/range_y", {snap.range_y[0], snap.range_y[1]});
        write_vec1d_double (f.id, base + "/meta/range_z", {snap.range_z[0], snap.range_z[1]});
    }

    if (H5Fflush(f.id, H5F_SCOPE_GLOBAL) < 0) throw_h5("H5Fflush failed");
    f.close();

    if (std::filesystem::exists(fin_path)) std::filesystem::remove(fin_path);
    std::filesystem::rename(tmp_path, fin_path);
}

} // namespace checkpoint
