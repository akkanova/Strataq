#include "strata/common/sample/row_runs.hpp"
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    // rows 0-2: slot 4's window; row 3: slot 4 at a gap; row 4: another history length; rows 5-6: slot 1
    const int owners[] = {4, 4, 4, 4, 4, 1, 1};
    const int64_t pos[] = {10, 11, 12, 20, 21, 7, 8};
    const int history[] = {64, 64, 64, 64, 32, 0, 0};
    std::vector<std::array<int64_t, 3>> runs;
    strata::sample::for_each_row_run(7, owners, pos, history, [&](int first, int end, uint64_t counter) {
        runs.push_back({first, end, (int64_t) counter});
    });
    strata::sample::for_each_row_run(0, nullptr, nullptr, nullptr, [&](int, int, uint64_t) { runs.push_back({}); });
    if (runs != std::vector<std::array<int64_t, 3>>{{0, 3, 10}, {3, 4, 20}, {4, 5, 21}, {5, 7, 7}}) {
        std::fputs("row_runs regression: wrong runs (first, end, counter)\n", stderr);
        return 1;
    }
    // Compare against Qwen's original adjacent-row loop, including repeated and decreasing positions.
    for (unsigned pattern = 0; pattern < 65536; ++pattern) {
        int owner[4], length[4];
        int64_t position[4];
        for (int r = 0; r < 4; ++r) {
            const unsigned bits = (pattern >> (4 * r)) & 15;
            owner[r] = bits & 1;
            length[r] = (bits >> 1) & 1;
            position[r] = bits >> 2;
        }
        std::vector<std::array<int64_t, 3>> expected, actual;
        for (int first = 0; first < 4;) {
            int end = first + 1;
            while (end < 4 && owner[end] == owner[first] && position[end] == position[end - 1] + 1 &&
                   length[end] == length[first])
                ++end;
            expected.push_back({first, end, position[first]});
            first = end;
        }
        strata::sample::for_each_row_run(4, owner, position, length,
                                        [&](int first, int end, uint64_t counter) {
            actual.push_back({first, end, (int64_t) counter});
        });
        if (actual != expected) {
            std::fputs("row_runs regression: grouping differs from Qwen's original loop\n", stderr);
            return 1;
        }
    }
    const int64_t large_pos[] = {std::numeric_limits<int64_t>::max() - 1, std::numeric_limits<int64_t>::max()};
    bool counters_match = true;
    strata::sample::for_each_row_run(2, owners, large_pos, history, [&](int first, int end, uint64_t counter) {
        for (int r = first; r < end; ++r)
            counters_match &= counter + (uint64_t) (r - first) == strata::sample::get_row_counter(large_pos[r]);
    });
    if (!counters_match || strata::sample::get_row_counter(0) != 0) {
        std::fputs("row_runs regression: grouped counters differ from individual rows\n", stderr);
        return 1;
    }
    std::puts("row_runs_test: passed");
}
