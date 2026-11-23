// src/common/state/conversation_block_tables.cpp - a conversation's block tables in a page pool
// (conversation_block_tables.hpp).
#include "strata/common/state/conversation_block_tables.hpp"

#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"

#include <algorithm>

namespace strata::core {

bool ConversationBlockTables::attach(PagePool& pool, ConversationId id, int device,
                                     const std::vector<TableSize>& sizes, uint64_t origin, std::string& err) {
    if (const std::string why = pool.grow_conversation(id, 0); !why.empty()) {
        err = "the page pool: " + why;
        return false;
    }
    if (device != device_ || sizes != sizes_)
        free_tables();
    const OnDevice on(device);
    device_ = device;
    sizes_ = sizes;
    origin_ = origin;
    tables_.resize(sizes.size(), nullptr);
    blocks_.resize(sizes.size());
    for (size_t r = 0; r < sizes.size(); ++r) {
        blocks_[r].clear();
        blocks_[r].reserve((size_t) sizes[r].entries);
        const size_t bytes = (size_t) sizes[r].entries * sizeof(int64_t);
        if (bytes > 0 && tables_[r] == nullptr &&
            (tables_[r] = (int64_t*) gpu_alloc_zeroed(bytes, "Conversation page tables")) == nullptr) {
            free_tables();
            pool.remove_conversation(id);
            err = "the page pool: no room for the conversation's block tables";
            return false;
        }
    }
    pool_ = &pool;
    id_ = id;
    return true;
}

bool ConversationBlockTables::grow(int64_t tokens, std::string& err) {
    if (const std::string why = pool_->grow_conversation(id_, tokens); !why.empty()) {
        err = "the conversation's memory is full: " + why;
        return false;
    }
    return write_tables(true, err);
}

bool ConversationBlockTables::refresh(std::string& err) {
    for (std::vector<uint8_t*>& at : blocks_)
        at.clear();
    return write_tables(false, err);
}

void ConversationBlockTables::truncate(int64_t tokens) {
    pool_->truncate_conversation(id_, tokens);
    for (size_t r = 0; r < blocks_.size(); ++r)
        blocks_[r].resize(std::min(blocks_[r].size(), pool_->conversation_blocks(id_, r).size()));
}

void ConversationBlockTables::release() {
    pool_->remove_conversation(id_);
    pool_ = nullptr;
}

bool ConversationBlockTables::has_blocks_for(int64_t tokens) const {
    if (!is_ready())
        return false;
    for (size_t r = 0; r < tables_.size(); ++r)
        if (tables_[r] != nullptr && pool_->region_blocks(r, tokens) > blocks_[r].size())
            return false;
    return true;
}

uint64_t ConversationBlockTables::bytes() const {
    uint64_t bytes = 0;
    for (size_t r = 0; is_ready() && r < pool_->layout().regions.size(); ++r)
        bytes += pool_->conversation_blocks(id_, r).size() * pool_->block_bytes(r);
    return bytes;
}

void ConversationBlockTables::free_tables() {
    const OnDevice on(device_);
    for (int64_t* t : tables_)
        gpu_free(t);
    tables_.clear();
}

bool ConversationBlockTables::write_tables(bool zero_new, std::string& err) {
    const OnDevice on(device_);
    bool added = false, ok = true;
    for (size_t r = 0; ok && r < tables_.size(); ++r) {
        const std::vector<PageId>& runs = pool_->conversation_blocks(id_, r);
        std::vector<uint8_t*>& at = blocks_[r];
        const size_t had = at.size();
        if (tables_[r] == nullptr || runs.size() == had)
            continue;
        for (size_t i = had; i < runs.size(); ++i) {
            at.push_back(pool_->page_address(runs[i]));
            ok = ok && (!zero_new || gpu_memset_async(at.back(), 0, pool_->block_bytes(r), nullptr));
        }
        const uint64_t per_block = sizes_[r].units_per_block, unit_bytes = pool_->layout().regions[r].unit_bytes;
        const int64_t entries = std::min((int64_t) (at.size() * per_block), sizes_[r].entries);
        const int64_t first = std::min((int64_t) (had * per_block), entries);
        std::vector<int64_t> table;
        for (int64_t e = first; e < entries; ++e)
            table.push_back((int64_t) ((uintptr_t) at[(size_t) e / per_block] + e % per_block * unit_bytes - origin_));
        ok = ok && (table.empty() || gpu_copy(tables_[r] + first, table.data(), table.size() * sizeof(int64_t)));
        if (!ok)
            at.resize(had);
        added = true;
    }
    // the kernels that read the tables and pages run on non-blocking streams, which do not wait for these copies
    if (ok && (!added || gpu_stream_synchronize(nullptr)))
        return true;
    err = "writing the conversation's new blocks failed";
    return false;
}

}  // namespace strata::core
