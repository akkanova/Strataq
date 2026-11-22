// How a step's sampled rows are seeded and grouped into sampler calls.
#pragma once

#include "strata/common/spec/step.hpp"

#include <cstdint>
#include <tuple>

namespace strata::sample {

/// Gets the Philox counter for a nonnegative input position, independently of earlier draws and batch layout.
/// Speculative acceptance and residual sampling may use different random streams from a plain draw.
inline uint64_t get_row_counter(int64_t pos) {
    return (uint64_t) pos;
}

/// Calls `draw(first, end, counter)` in row order for each run [first, end) of the n rows that one sampler call can
/// draw: rows of one owner (conversation or slot) `owners[r]`, at consecutive input positions `pos[r]`, with penalty
/// histories of one length `history[r]`. `counter` is the first row's get_row_counter; row r's is counter + r - first.
template <class Draw>
void for_each_row_run(int n, const int* owners, const int64_t* pos, const int* history, Draw&& draw) {
    strata::spec::for_each_window(n, [&](int r) { return std::tuple(owners[r], pos[r] - r, history[r]); },
                                  [&](int first, int end) { draw(first, end, get_row_counter(pos[first])); });
}

/// Reports whether row r has another input in its verify window, independently of penalty-history run boundaries.
inline bool has_next_window_row(int n, const int* owners, const int64_t* pos, int r) {
    return r + 1 < n && owners[r + 1] == owners[r] && pos[r + 1] > pos[r] && pos[r + 1] - 1 == pos[r];
}

}  // namespace strata::sample
