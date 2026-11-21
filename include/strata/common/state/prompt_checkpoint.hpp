#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// Finds where history ends before the next answer, excluding a temporary effort turn when present.
inline int64_t find_prompt_turn_boundary(const std::vector<int64_t>& ids, int64_t from, int64_t turn_token,
                                         int64_t tail_role_token) {
    if (turn_token < 0)
        return -1;
    int64_t turn = -1;
    for (int64_t i = (int64_t) ids.size() - 1; i > from; --i)
        if (ids[(std::size_t) i] == turn_token) {
            turn = i;
            break;
        }
    if (turn > 0 && tail_role_token >= 0)
        for (int64_t i = turn - 1; i > from; --i)
            if (ids[(std::size_t) i] == turn_token) {
                if (ids[(std::size_t) i + 1] == tail_role_token)
                    turn = i;
                break;
            }
    return turn;
}

/// Finds the end of a sufficiently long opening system turn for root-pinned checkpoint retention.
inline int64_t find_prompt_root_boundary(const std::vector<int64_t>& ids, int64_t turn, int64_t turn_token,
                                         int64_t minimum) {
    if (turn_token >= 0 && minimum > 0)
        for (int64_t i = 1; i < turn; ++i)
            if (ids[(std::size_t) i] == turn_token)
                return i >= minimum ? i : -1;
    return -1;
}

/// Says where a model wants its prompt reads to keep checkpoints of their own, beside the restore points a request
/// asks for (Qwen3.8's --turn-token, --tail-role-token, --prompt-cache-root and --prompt-cache-every).
/// PagedModelFacts carries it to the paged engine's planner.
struct CheckpointPlacement {
    int64_t turn_token = -1;      ///< the token that opens a chat turn; -1: no turn or root checkpoints
    int64_t tail_role_token = -1; ///< the role token of a temporary last turn that the turn checkpoint goes before
    int64_t root_minimum = 0;     ///< the fewest tokens an opening system turn needs to be kept as the root; 0: none
    int64_t every = 0;            ///< the spacing of checkpoints before the last turn; 0: none
};

/// Finds where a read of prompt `ids` from position `from` keeps checkpoints, in position order: the root, the turn
/// boundary and the periodic points before that turn (the prompt's last token is read with its answer, not before).
inline std::vector<int64_t> find_prompt_checkpoints(const CheckpointPlacement& p, const std::vector<int64_t>& ids,
                                                    int64_t from) {
    const int64_t end = (int64_t) ids.size() - 1;
    const int64_t turn = find_prompt_turn_boundary(ids, 0, p.turn_token, p.tail_role_token);
    const int64_t root = find_prompt_root_boundary(ids, turn, p.turn_token, p.root_minimum);
    std::vector<int64_t> cuts;
    for (const int64_t point : {root, turn})
        if (point > from && point < end)
            cuts.push_back(point);
    if (p.every > 0)
        for (int64_t point = (from / p.every + 1) * p.every; point <= end && (turn < 0 || point < turn);
             point += p.every)
            cuts.push_back(point);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    return cuts;
}

}  // namespace strata::core
