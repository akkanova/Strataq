// Builds the paged engine's per-row sampler histories.
#pragma once

#include "strata/common/kernels/sampler.hpp"
#include "strata/common/spec/step.hpp"

#include <vector>

namespace strata::serve {

/// Builds each row's recent tokens from its conversation and its window's preceding guesses.
inline std::vector<std::vector<int32_t>> get_penalty_histories(
    const std::vector<strata::spec::StepRow>& rows, const std::vector<const std::vector<int32_t>*>& tails,
    const std::vector<int>& lengths) {
    std::vector<std::vector<int32_t>> result(rows.size());
    strata::spec::for_each_window(rows, [&](size_t first, size_t end) {
        const int h = lengths[(size_t) rows[first].session];
        if (h <= 0)
            return;
        std::vector<int32_t> tokens, staged((end - first) * (size_t) h);
        for (size_t r = first; r < end; ++r)
            tokens.push_back(rows[r].token);
        const auto& tail = *tails[first];
        strata::kernels::penalty_rows(tail.data(), (int64_t) tail.size(), tokens.data(), (int) tokens.size(), h,
                                      staged.data());
        for (size_t r = first; r < end; ++r)
            result[r].assign(staged.begin() + (r - first) * h, staged.begin() + (r - first + 1) * h);
    });
    return result;
}

}  // namespace strata::serve
