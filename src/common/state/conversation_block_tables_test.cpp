// A conversation's block tables (conversation_block_tables.hpp) over a page pool in GPU memory.  Needs a GPU.
#include "strata/common/state/conversation_block_tables.hpp"

#include "strata/base/gpu/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

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

constexpr uint64_t kUnitBytes = 16, kOrigin = 4096;

/// Tells whether region 1's table holds its first `entries` entries: each unit's address less kOrigin.
bool is_table_right(const ConversationBlockTables& t, uint64_t per_block, int64_t entries) {
    std::vector<int64_t> table((size_t) entries);
    check(gpu_copy(table.data(), t.table(1), table.size() * sizeof(int64_t)), "the table reads back");
    for (int64_t e = 0; e < entries; ++e)
        if (table[(size_t) e] != (int64_t) ((uintptr_t) t.blocks(1).at((size_t) e / per_block) +
                                            e % per_block * kUnitBytes - kOrigin))
            return false;
    return true;
}

/// Gets the first byte of block `b` of region 1.
uint8_t read_byte(const ConversationBlockTables& t, size_t b) {
    uint8_t byte = 0;
    check(gpu_copy(&byte, t.blocks(1).at(b), 1), "a block reads back");
    return byte;
}

/// Sets the first byte of block `b` of region 1.
bool write_byte(const ConversationBlockTables& t, size_t b, int value) {
    return gpu_memset_async(t.blocks(1).at(b), value, 1, nullptr) && gpu_stream_synchronize(nullptr);
}
}  // namespace

int main() {
    StateLayout layout;
    layout.add(RegionKind::fixed, 100);
    layout.add(RegionKind::per_token, kUnitBytes);
    PagePoolConfig config;
    config.page_bytes = 64;
    config.chunk_bytes = 64 * 64;
    std::vector<void*> chunks;
    PagePoolHooks hooks;
    hooks.alloc = [&chunks](size_t bytes) { return chunks.emplace_back(gpu_alloc(bytes)); };
    hooks.free = [](void* p) { gpu_free(p); };
    hooks.copy = [&chunks, &config](PageId to, PageId from, uint32_t pages) {
        const uint64_t per_chunk = config.chunk_bytes / config.page_bytes;
        const auto at = [&](PageId p) {
            return (uint8_t*) chunks.at(p / per_chunk) + p % per_chunk * config.page_bytes;
        };
        return gpu_copy(at(to), at(from), pages * config.page_bytes) && gpu_stream_synchronize(nullptr);
    };
    PagePool pool(layout, config, hooks);
    pool.resize(config.chunk_bytes);
    if (chunks.empty() || chunks[0] == nullptr) {
        std::printf("SKIP: no GPU\n");
        return 0;
    }
    check(gpu_memset_async(chunks[0], 0xEE, config.chunk_bytes, nullptr) && gpu_stream_synchronize(nullptr),
          "the pool's pages fill");
    const uint64_t per_block = pool.block_bytes(1) / kUnitBytes;   // a unit per token
    const int64_t cap = (int64_t) (5 * per_block) - 1, one = (int64_t) per_block, two = 2 * one;
    const std::vector<ConversationBlockTables::TableSize> sizes = {{}, {cap, per_block}};
    ConversationBlockTables t;
    std::string err;

    check(t.attach(pool, 1, 0, sizes, kOrigin, err) && t.id() == 1 && !t.table(0) && t.table(1),
          "attach makes a table per growing region");
    check(t.bytes() == pool.block_bytes(0), "attach holds the fixed region");
    int64_t* const table = t.table(1);
    std::vector<int64_t> zeros((size_t) cap), host((size_t) cap);
    check(gpu_copy(host.data(), table, host.size() * sizeof(int64_t)) && host == zeros, "a new table is zeroed");
    uint8_t* const* const host_blocks = t.blocks(1).data();

    check(t.grow(two, err) && is_table_right(t, per_block, two) && read_byte(t, 0) == 0 && read_byte(t, 1) == 0,
          "a grow zeroes the new blocks and uploads their entries");
    check(t.has_blocks_for(two) && !t.has_blocks_for(two + 1), "has_blocks_for counts the written blocks");
    const int64_t sentinel = 123;
    check(gpu_copy(table, &sentinel, sizeof sentinel), "an old entry is marked");
    check(write_byte(t, 0, 0x5A) && t.grow(6 * one, err) && read_byte(t, 0) == 0x5A && read_byte(t, 5) == 0,
          "a second grow zeroes only its new blocks");
    int64_t kept = 0;
    check(gpu_copy(&kept, table, sizeof kept) && kept == sentinel, "growth uploads only new entries");
    check(t.refresh(err), "refresh restores the marked entry");
    check(is_table_right(t, per_block, cap) && t.blocks(1).data() == host_blocks, "entries stop at the table's size");
    check(t.bytes() == pool.block_bytes(0) + 6 * pool.block_bytes(1), "bytes counts every block");

    t.truncate(one);
    check(t.blocks(1).size() == 1 && t.has_blocks_for(one) && !t.has_blocks_for(two), "a truncate drops blocks");
    check(t.grow(two, err) && is_table_right(t, per_block, two) && read_byte(t, 1) == 0, "a grow after a truncate");

    check(write_byte(t, 1, 0x77) && gpu_copy(table, &table, sizeof table), "a block and an entry are spoiled");
    check(t.refresh(err) && is_table_right(t, per_block, two) && read_byte(t, 1) == 0x77,
          "a refresh rewrites every entry and zeroes nothing");

    t.release();
    check(!t.is_ready() && t.bytes() == 0, "a release gives back every block");
    check(t.attach(pool, 2, 0, sizes, kOrigin, err) && t.table(1) == table && t.blocks(1).empty() &&
          t.blocks(1).data() == host_blocks, "a new attach of the same sizes keeps the table and host storage");
    check(!t.has_blocks_for(1) && t.grow(two, err) && is_table_right(t, per_block, two), "a reused table");
    t.release();
    check(t.attach(pool, 3, 0, {{}, {cap + 1, 1}}, kOrigin, err) && t.table(1),
          "a new attach of other sizes makes a table");
    check(t.grow(two, err) && is_table_right(t, 1, 2), "DeepSeek tables hold one address per pool block");
    t.release();
    check(pool.resize(2 * config.chunk_bytes).refusal.empty(), "the pool grows to two chunks");
    check(pool.grow_conversation(4, 248).empty(), "another conversation occupies the first chunk");
    check(t.attach(pool, 5, 0, sizes, kOrigin, err) && t.grow(two, err), "a conversation uses the second chunk");
    uint8_t* const before = t.blocks(1).at(0);
    check(write_byte(t, 0, 0x33), "the moving block holds data");
    pool.remove_conversation(4);
    check(pool.resize(config.chunk_bytes).refusal.empty() && t.refresh(err), "a shrink relocates live blocks");
    check(t.blocks(1).at(0) != before && read_byte(t, 0) == 0x33 && is_table_right(t, per_block, two),
          "refresh publishes relocated addresses and preserves data");
    std::printf("conversation_block_tables_test: %d checks passed\n", checks);
    return 0;
}
