#pragma once

#include <array>
#include <string>
#include <vector>

#include "Checkpoint.hpp"

namespace checkpoint {

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

    const AlphaSnapshot* alpha_snapshot
);

} // namespace checkpoint
