// include/strata/common/state/conversation_block_tables.hpp - the device tables through which a model's kernels find
// one conversation's blocks in a page pool (page_pool.hpp).
#pragma once

#include "strata/common/state/page_pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One conversation's place in a page pool and, per growing region, its blocks' addresses on the host and a device
/// table for the kernels whose entries are the addresses of units of those blocks, less an origin.  Used by DeepSeek
/// V4.1's paged Part (an entry per block) and Qwen3.8's ConversationPages (an entry per K/V unit).
class ConversationBlockTables {
public:
    /// The size of one region's device table.
    struct TableSize {
        int64_t entries = 0;          ///< Most entries it holds; 0: no table (a fixed region).
        uint64_t units_per_block = 1; ///< Entries one of the pool's blocks gives, the region's unit_bytes apart.
        bool operator==(const TableSize&) const = default;
    };

    ConversationBlockTables() = default;
    ~ConversationBlockTables() { free_tables(); }
    ConversationBlockTables(const ConversationBlockTables&) = delete;
    ConversationBlockTables& operator=(const ConversationBlockTables&) = delete;

    /// Adds conversation `id` to `pool` with its fixed regions, and gives it device tables of `sizes` (one per region
    /// of the pool's layout) on GPU `device`, whose entries are addresses less `origin`.  New tables are zeroed; an
    /// earlier attach's of the same sizes are kept as they are, so that the kernels' pointers stay valid.  False (and
    /// `err`) when the pool or the GPU has no room; the pool then lacks the conversation.
    /// Callers must drain consumers before changing tables and reserve before reading entries; unused entries may
    /// retain addresses of returned pages. The pool must outlive this attachment.
    bool attach(PagePool& pool, ConversationId id, int device, const std::vector<TableSize>& sizes, uint64_t origin,
                std::string& err);
    bool is_ready() const { return pool_ != nullptr; }
    /// Gives the conversation the blocks for `tokens` tokens, zeroes the new blocks and uploads their entries.
    bool grow(int64_t tokens, std::string& err);
    /// Writes every entry again after a shrink of the pool moved the blocks (their bytes moved with them).
    bool refresh(std::string& err);
    /// Gives back the growing regions' blocks past the first `tokens` tokens.
    void truncate(int64_t tokens);
    /// Gives back every block, the fixed regions' too; the device tables stay for the next attach.
    void release();
    /// Tells whether the tables already hold every block of a conversation of `tokens` tokens.
    bool has_blocks_for(int64_t tokens) const;
    /// Gets the bytes of the pool's pages the conversation holds (0 before attach).
    uint64_t bytes() const;

    ConversationId id() const { return id_; }
    const PagePool& pool() const { return *pool_; }
    /// Gets the device table of region `r` (null for a region without one).
    int64_t* table(size_t r) const { return r < tables_.size() ? tables_[r] : nullptr; }
    /// Gets region `r`'s block addresses. For growth within TableSize::entries, their storage stays put until attach.
    const std::vector<uint8_t*>& blocks(size_t r) const { return blocks_.at(r); }
    /// Gets the address of fixed region `r`'s block.
    uint8_t* get_fixed_address(size_t r) const {
        return pool_->page_address(pool_->conversation_blocks(id_, r).at(0));
    }

private:
    /// Uploads the entries of the pool's blocks that blocks_ lacks, zeroing those blocks first with `zero_new`
    /// (their pages held other data).
    bool write_tables(bool zero_new, std::string& err);
    /// Frees device storage on its owning GPU, keeping the host block lists.
    void free_tables();

    PagePool* pool_ = nullptr;
    ConversationId id_ = 0;
    int device_ = 0;
    uint64_t origin_ = 0;
    std::vector<TableSize> sizes_;                ///< Per region of the layout.
    std::vector<int64_t*> tables_;                ///< Per region: its device table, or null.
    std::vector<std::vector<uint8_t*>> blocks_;   ///< Per region: its blocks whose entries the table holds.
};

}  // namespace strata::core
