// The RAM's page pool (ram_experts.hpp) over an in-memory "SSD": it fills with the ranking's first experts, hands out
// their bytes and reads the others into its staging ring (once each, also after prefetch), gives experts' pages to a
// paused conversation and takes them back after it leaves, copies in an expert a GPU evicts, keeps an expert a step
// reads from the SSD in place of one used before the step's layer, replaces one expert by another in place, and takes
// a new ranking (a layer split's cut moved).
#include "strata/common/experts/ram_experts.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

constexpr uint64_t kPage = 16384;
/// Layers 0 and 2 have experts of one page, layer 1 of two.
const std::vector<uint64_t> kBlobs = {kPage, 2 * kPage, kPage};
constexpr int64_t kExperts = 8;

/// Gets the byte at `i` of expert (`layer`, `expert`).
uint8_t get_byte(int64_t layer, int64_t expert, uint64_t i) { return (uint8_t) (layer * 31 + expert * 7 + i); }

/// The experts' bytes as the SSD would give them, counting the reads.
struct Ssd : ExpertSource {
    int reads = 0, batches = 0;
    bool fail = false;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override {
        ++batches;
        for (size_t i = 0; i < n; ++i)
            if (!copy_blob(layers[i], experts[i], dst[i]))
                return false;
        return true;
    }
    std::vector<uint8_t> buffer = std::vector<uint8_t>(2 * kPage);
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        (void) copy_blob(layer, expert, buffer.data());
        return buffer.data();
    }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        ++reads;
        if (fail)
            return false;
        for (uint64_t i = 0; i < kBlobs[(size_t) layer]; ++i)
            dst[i] = get_byte(layer, expert, i);
        return true;
    }
};

/// Tells whether `bytes` are expert (`layer`, `expert`)'s.
bool is_expert(const uint8_t* bytes, int64_t layer, int64_t expert) {
    for (uint64_t i = 0; i < kBlobs[(size_t) layer]; ++i)
        if (bytes == nullptr || bytes[i] != get_byte(layer, expert, i))
            return false;
    return true;
}
}  // namespace

int main() {
    TrafficCount count;
    std::thread first([&] {
        for (int i = 0; i < 10000; ++i)
            ++count;
    });
    std::thread second([&] {
        for (int i = 0; i < 10000; ++i)
            ++count;
    });
    first.join();
    second.join();
    const TrafficCount snapshot = count;
    TrafficCount assigned;
    assigned = count;
    ++count;
    check(snapshot == 20000 && assigned == 20000 && count == 20001,
          "concurrent increments are retained and copies are independent snapshots");
    count += UINT64_MAX;
    check(count == 20000, "overflow wraps as an unsigned total");

    Ssd ssd;
    StateLayout layout;   // a conversation of a 3-page state and 1 page per 4 tokens
    layout.add(RegionKind::fixed, 3 * kPage);
    layout.add(RegionKind::per_block, kPage, 1, 4);
    std::vector<ExpertKey> ranking;
    for (int32_t e = 0; e < kExperts; ++e)
        for (int32_t l = 0; l < 3; ++l)
            ranking.push_back({l, e});
    PagePoolConfig config;
    config.page_bytes = kPage;
    config.hysteresis = 0;
    {
        Ssd staged_ssd;
        RamExperts staged;
        PagePoolConfig chunks = config;
        chunks.chunk_bytes = 4 * kPage;
        std::string err;
        const uint64_t asked = 16 * kPage;
        check(staged.open(staged_ssd, asked, 2, layout, chunks, kBlobs, kExperts, ranking, err), err.c_str());
        check(staged.get_asked_bytes() == asked && staged.pages().pool().report().total.pages == 4,
              "opening retains the requested size while allocating only one chunk");
        check(staged.resize(asked).refusal.empty() && staged.pages().pool().report().total.pages == 12,
              "deferred growth uses the requested size including the staging ring");
        uint64_t wait = 0;
        check(staged.refill(16, wait, err) == 9 && staged.holds(1, 2),
              "deferred growth admits the full ranking beyond the initial chunk");
    }
    RamExperts ram;
    std::string err;
    // room for 12 pages: experts 0-2 of every layer (12 pages)
    check(ram.open(ssd, 12 * kPage + 2 * 2 * kPage, 2, layout, config, kBlobs, kExperts, ranking, err), err.c_str());
    check(ssd.reads == 0 && ram.pages().is_refill_wanted(), "open performs no expert reads");
    uint64_t startup_wait = 0;
    check(ram.refill(16, startup_wait, err) == 9, "refill brings in the experts that fit");
    check(ssd.batches == 1, "a refill submits its experts in one SSD read");
    const ExpertTraffic& traffic = ram.pages().traffic();
    check(traffic.loads == 9 && traffic.ssd_reads == 1 && traffic.ssd_bytes == 12 * kPage,
          "the refill's experts and its SSD read are counted");
    check(ram.holds(1, 2) && !ram.holds(0, 3) && ram.pages().pool().report().experts.pages == 12,
          "the pool holds the ranking's first experts");
    {
        Ssd colder_ssd;
        RamExperts colder;
        check(colder.open(colder_ssd, 12 * kPage + 2 * 2 * kPage, 2, layout, config, kBlobs, kExperts, ranking, err),
              err.c_str());
        uint64_t ms = 0;
        check(colder.refill(16, ms, err) == 9, "the colder pool fills on request");
        PagePool& pool = colder.pages().pool();
        check(pool.grow_conversation(9, 8).empty(), "a paused conversation evicts experts");
        pool.remove_conversation(9);
        for (int32_t e = 3; e < 6; ++e)   // colder experts than those it evicted take the pages it left
            for (int32_t l = 0; l < 3; ++l)
                (void) colder.pages().admit(l, e);
        check(pool.find_expert({0, 3}) != no_page && colder.pages().is_refill_wanted() &&
                  colder.refill(16, ms, err) == 0 && pool.find_expert({0, 3}) != no_page &&
                  !colder.pages().is_refill_wanted(),
              "a refill evicts no expert: it waits for free pages");
    }
    {
        Ssd full_ssd;
        RamExperts full;
        check(full.open(full_ssd, 12 * kPage + 2 * 2 * kPage, 2, layout, config, kBlobs, kExperts, ranking, err),
              err.c_str());
        uint64_t ms = 0;
        check(full.refill(16, ms, err) == 9 && full.pages().pool().report().free.pages == 0, "the full pool fills");
        std::vector<float> usage(3 * kExperts, 1.0f);   // the adaptive tier's routing counts
        usage[2 * kExperts + 0] = 0.2f;                  // (2, 0): the least used of the one-page experts held
        usage[0 * kExperts + 5] = 0.1f;
        std::vector<uint8_t> gpu(kPage);
        for (uint64_t i = 0; i < kPage; ++i)
            gpu[i] = get_byte(0, 5, i);
        check(!full.keep_copy({0, 5}, gpu.data(), usage) && !full.holds(0, 5) && full.holds(2, 0),
              "a full pool keeps no copy of an expert used less than every one of its size it holds");
        usage[0 * kExperts + 5] = 0.2f;
        check(!full.keep_copy({0, 5}, gpu.data(), usage) && full.holds(2, 0),
              "equal routing counts do not displace a RAM expert");
        check(!full.keep_copy({0, 5}, gpu.data(), {}) && full.holds(2, 0),
              "missing routing counts count as zero and do not displace an expert");
        usage[0 * kExperts + 5] = 0.3f;
        check(full.keep_copy({0, 5}, gpu.data(), usage) && full.holds(0, 5) && !full.holds(2, 0) &&
                  is_expert(full.blob(0, 5), 0, 5) && full_ssd.reads == 9,
              "a copy takes the pages of the least used expert of its size when that is used less");
    }
    {
        Ssd later_ssd;
        RamExperts later;
        check(later.open(later_ssd, 12 * kPage + 4 * kPage, 2, layout, config, kBlobs,
                         kExperts, ranking, err), err.c_str());
        check(later_ssd.reads == 0, "the startup pool is empty");
        std::vector<int32_t> residency(3 * kExperts, -1);
        residency[0] = 0;
        later.pages().stop_wanting_if([&](ExpertKey k) {
            return residency[(size_t) (k.layer * kExperts + k.expert)] >= 0;
        });
        uint64_t ms = 0;
        check(later.refill(1, ms, err) == 1 && later.holds(1, 0) && !later.holds(0, 0),
              "the RAM's fill skips an expert a GPU holds");
        check(!later.holds(1, 5) && later.pages().replace({1, 0}, {1, 5}, err) == 1 && later.holds(1, 0) &&
                  later.holds(1, 5) && is_expert(later.blob(1, 5), 1, 5),
              "while the pool has free pages, an expert brought in pushes none out");
        check(is_expert(later.blob(2, 7), 2, 7), "a conversation can read before refill completes");
        const uint64_t loads_before_failure = later.pages().traffic().loads;
        later_ssd.fail = true;
        check(later.refill(2, ms, err) == -1 && !later.holds(2, 0) && !later.holds(0, 1) &&
                  later.pages().traffic().loads == loads_before_failure,
              "a failed SSD batch publishes none of its experts");
        later_ssd.fail = false;
        check(later.pages().is_refill_wanted() && later.refill(2, ms, err) == 2 && later.holds(2, 0) &&
                  later.holds(0, 1), "a failed refill keeps its experts eligible for retry");
    }
    check(is_expert(ram.blob(1, 2), 1, 2) && !ram.transient(1, 2), "an expert in the pool is its bytes there");
    const int filled = ssd.reads;
    check(is_expert(ram.blob(2, 5), 2, 5) && ram.transient(2, 5) && ssd.reads == filled + 1,
          "an expert outside the pool is read into the ring");
    const int64_t wanted[] = {3, 4, 5};
    ram.prefetch(0, wanted, 3);   // two buffers: the third read takes the first one again
    check(ssd.reads == filled + 4 && is_expert(ram.blob(0, 5), 0, 5) && ssd.reads == filled + 4,
          "a prefetched expert is not read again");
    check(is_expert(ram.blob(0, 3), 0, 3) && ssd.reads == filled + 5, "a buffer the ring took back is read again");
    std::vector<uint8_t> out(2 * kPage);
    check(ram.copy_blob(1, 0, out.data()) && is_expert(out.data(), 1, 0) && ram.copy_blob(1, 6, out.data()) &&
              is_expert(out.data(), 1, 6),
          "copy_blob gives the pool's and the SSD's experts");
    check(traffic.hits == 2 && traffic.misses == 4, "each read counts a hit or, outside the pool, a miss");
    {
        const ExpertKey keys[] = {{1, 0}, {1, 6}, {2, 6}};   // one the pool holds, two it does not
        const uint64_t bytes[] = {kBlobs[1], kBlobs[1], kBlobs[2]};
        const int reads = ssd.reads, batches = ssd.batches;
        check(ram.use_blobs(keys, bytes, 3, [&](const uint8_t* const* from) {
                  return is_expert(from[0], 1, 0) && is_expert(from[1], 1, 6) && is_expert(from[2], 2, 6);
              }) && ssd.reads == reads + 2 && ssd.batches == batches + 1 &&
                  !ram.holds(1, 6) && !ram.holds(2, 6),
              "use_blobs gives held experts and reads the others in one batch");
        ssd.fail = true;
        check(!ram.use_blobs(keys, bytes, 3, [](const uint8_t* const*) {
                  check(false, "a failed read must not call the consumer");
                  return true;
              }), "use_blobs fails when the SSD read fails");
        ssd.fail = false;
    }

    {
        RamExperts trunk;
        check(trunk.open(ssd, 4 * kPage, 1, layout, config, {kBlobs[0]}, kExperts, {{0, 0}}, err), err.c_str());
        const ExpertKey key{2, 6};
        const uint64_t bytes = kBlobs[2];
        check(trunk.use_blobs(&key, &bytes, 1, [](const uint8_t* const* from) {
                  return is_expert(from[0], 2, 6);
              }), "use_blobs reads a drafter layer past the RAM pool's layers");
        check(!trunk.use_blobs(&key, &bytes, 1, [](const uint8_t* const*) { return false; }),
              "use_blobs reports a consumer failure");
    }

    PagePool& pool = ram.pages().pool();
    check(pool.grow_conversation(9, 8).empty(), "a paused conversation takes pages");
    check(pool.report().experts.pages <= 12 - 5 && traffic.evictions > 0, "it took experts' pages");
    check(is_expert(ram.blob(2, 2), 2, 2) && is_expert(ram.blob(1, 7), 1, 7), "an evicted expert is read from the SSD");
    std::vector<uint8_t> gpu(kPage);
    for (uint64_t i = 0; i < kPage; ++i)
        gpu[i] = get_byte(0, 7, i);
    pool.remove_conversation(9);
    std::vector<float> usage(3 * kExperts, 0.0f);   // the adaptive tier's routing counts
    usage[7] = 1.0f;
    ram.keep_copy({0, 7}, gpu.data(), usage);   // a GPU's evicted expert, used more than any the pool holds
    check(ram.holds(0, 7) && is_expert(ram.blob(0, 7), 0, 7),
          "an expert a GPU evicts is copied in when there is room or a less used expert of its size");
    uint64_t wait_ms = 0;
    check(ram.pages().is_refill_wanted() && ram.refill(16, wait_ms, err) >= 1, "the evicted experts come back");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));   // the pool's clock counts milliseconds
    ram.begin_layer(1, nullptr, 0);
    check(!ram.holds(1, 5) && is_expert(ram.blob(1, 5), 1, 5) && ram.holds(1, 5),
          "an expert a step reads from the SSD takes the place of one used before the layer");
    int32_t gone = 0, in = 0;   // one expert of layer 2 the pool holds, one it does not
    while (!ram.holds(2, gone))
        ++gone;
    while (ram.holds(2, in))
        ++in;
    const int before = ssd.reads;
    const ExpertTraffic was = traffic;
    check(ram.pages().replace({2, gone}, {2, in}, err) == 1 && ram.holds(2, in) && !ram.holds(2, gone) &&
              ssd.reads == before + 1 && is_expert(ram.blob(2, in), 2, in) && traffic.swaps == was.swaps + 1 &&
              traffic.loads == was.loads + 1,
          "an expert replaced in place leaves, and the other takes its pages with its own bytes");
    const auto check_bytes = [&] {
        for (int64_t l = 0; l < 3; ++l)
            for (int64_t e = 0; e < kExperts; ++e)
                check(!ram.holds(l, e) || is_expert(ram.blob(l, e), l, e), "every expert in the pool has its bytes");
    };
    check_bytes();

    // a second chunk: the experts that did not fit come in; a paused conversation in it moves when it is given back
    check(ram.resize(ram.ring_bytes() + 24 * kPage).refusal.empty() && pool.report().total.pages == 24,
          "growing adds a chunk");
    while (ram.pages().is_refill_wanted() && ram.refill(16, wait_ms, err) > 0) {
    }
    check(pool.report().experts.pages > 12, "the experts that did not fit come in");
    check(pool.grow_conversation(9, 8).empty(), "a conversation takes pages in the larger pool");
    std::vector<uint8_t*> blocks;
    for (size_t r = 0; r < 2; ++r)
        for (const PageId p : pool.conversation_blocks(9, r))
            blocks.push_back(pool.page_address(p));
    for (size_t i = 0; i < blocks.size(); ++i)
        std::memset(blocks[i], (int) (i + 1), kPage);
    pool.set_paused(9, true);
    check(ram.resize(ram.ring_bytes() + 12 * kPage).refusal.empty() && pool.report().total.pages == 12,
          "shrinking gives the chunk back");
    size_t i = 0;
    for (size_t r = 0; r < 2; ++r)
        for (const PageId p : pool.conversation_blocks(9, r))
            check(p < 12 && pool.page_address(p)[0] == (uint8_t) ++i && pool.page_address(p)[kPage - 1] == (uint8_t) i,
                  "the conversation's pages keep their bytes in the first chunk");
    check_bytes();

    // a new ranking of layer 1's experts only (a card that runs that layer alone now): the others leave
    pool.remove_conversation(9);
    std::vector<ExpertKey> layer1;
    for (int32_t e = 0; e < kExperts; ++e)
        layer1.push_back({1, e});
    ram.pages().set_ranking(layer1, 4);
    bool others_left = true;
    for (int32_t e = 0; e < kExperts; ++e)
        others_left = others_left && !ram.holds(0, e) && !ram.holds(2, e);
    check(others_left && !ram.holds(1, 7), "the experts outside the new ranking's first leave");
    while (ram.pages().is_refill_wanted() && ram.refill(16, wait_ms, err) > 0) {
    }
    check(ram.holds(1, 0) && ram.holds(1, 3) && !ram.holds(1, 4) && pool.report().experts.pages == 8,
          "the new ranking's first come in");
    check_bytes();
    const auto before_speed_test = traffic.get_counts();
    check(ram.time_ssd_reads(0) > 0 && traffic.get_counts() == before_speed_test,
          "the SSD speed probe leaves traffic counters unchanged");
    std::printf("ram_experts_test: %d checks passed\n", checks);
    return 0;
}
