// include/strata/common/experts/expert_traffic.hpp - counts of what an expert pool (ExpertPages, RamExperts) did
// since the engine started, for GET_MEMORY_USE's EXPERT_TRAFFIC line (memory_report.hpp) and the Monitor page.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// A running total that threads add to without ordering anything else (relaxed), so that counting costs one
/// atomic add; copies sample each counter independently. Totals wrap modulo 2^64.
/// Pools holding these counters stay movable (a deque of cards resizes).
class TrafficCount {
public:
    static_assert(std::atomic<uint64_t>::is_always_lock_free);

    TrafficCount() = default;
    TrafficCount(const TrafficCount& o) : n_(o) {}
    TrafficCount& operator=(const TrafficCount& o) {
        n_.store(o, std::memory_order_relaxed);
        return *this;
    }
    void operator+=(uint64_t k) { n_.fetch_add(k, std::memory_order_relaxed); }
    void operator++() { *this += 1; }
    void operator--() { n_.fetch_sub(1, std::memory_order_relaxed); }
    operator uint64_t() const { return n_.load(std::memory_order_relaxed); }

private:
    std::atomic<uint64_t> n_{0};
};

/// Running totals of one expert pool's work: experts found in it or not, copied into it, pushed out of it, and read
/// from the SSD for it.  The RAM's pool counts on the prompt path's threads too.  A GPU's hits and misses are decode
/// steps' routings (each expert once a step) to the experts it holds and to its layers' experts that no GPU holds,
/// which the paged engine counts.
struct ExpertTraffic {
    TrafficCount hits;        ///< uses of an expert the pool held
    TrafficCount misses;      ///< uses outside the pool; GPU misses include CPU computation and reads over PCIe
    TrafficCount loads;       ///< experts copied into the pool's pages
    TrafficCount evictions;   ///< experts whose pages went to conversations, prompts, a shrink or a new ranking
    TrafficCount swaps;       ///< experts that took a colder one's pages (the adaptive tier's moves)
    TrafficCount ssd_reads;   ///< batches of experts read from the SSD
    TrafficCount ssd_bytes;   ///< bytes of experts read from the SSD
    TrafficCount ssd_wait_us; ///< microseconds spent waiting for those reads

    /// Gets the counters as name/value pairs, in a fixed order, leaving out the SSD's for a pool that reads none.
    std::vector<std::pair<std::string, uint64_t>> get_counts() const {
        std::vector<std::pair<std::string, uint64_t>> counts{
            {"hits", hits}, {"misses", misses}, {"loads", loads}, {"evictions", evictions}, {"swaps", swaps}};
        if (ssd_reads > 0)
            counts.insert(counts.end(),
                          {{"ssd_reads", ssd_reads}, {"ssd_bytes", ssd_bytes}, {"ssd_wait_us", ssd_wait_us}});
        return counts;
    }
};

}  // namespace strata::core
