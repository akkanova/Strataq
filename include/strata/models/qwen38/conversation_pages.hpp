// include/strata/models/qwen38/conversation_pages.hpp - a Qwen3.8 conversation's state in a GPU page pool: the GDN
// layers' recurrent state in a fixed region and every QSA layer's K/V and pooled indexer keys in units of the pool's
// pages, which grow with the conversation, and with drafting the MTP drafter's K/V in units too
// (docs/MEMORY_MANAGER_PLAN.md, steps M2d, M2e, M2f and M4).
//
// The session the pages are for is carved with session_set_pool_state, so it holds neither; `init` points it at the
// pool's pages.  The small running state (the PLE history and the indexer's tail, dead key and block position, the
// first region of describe_conversation) stays in the session for now.
#pragma once

#include "strata/common/state/conversation_block_tables.hpp"
#include "strata/core/geometry.hpp"
#include "strata/models/qwen38/kernels/kv_units.hpp"
#include "strata/models/qwen38/session.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen38 {

/// One conversation's pages in a pool of the current GPU's memory, which it shares with the experts (ExpertPages) and
/// with strata-q38's other conversations.
class ConversationPages {
public:
    /// Gets the regions a conversation of session `ss` keeps in a page pool: the GDN state of the session's layers,
    /// then each QSA layer's units and, with `drafter` supplied,
    /// the MTP drafter's units. The session may be a layer split's card's, carved for a range of the layers.  False (and
    /// `err`) when `ss` was carved with its state, or its range lacks a GDN or a QSA layer.
    static bool get_layout(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss,
                           const strata::core::QsaState* drafter, strata::core::StateLayout& layout,
                           std::string& err);
    /// Gives the conversation `id` of `pool` (a pool made with get_layout's regions; it must outlive this object) its
    /// fixed region and points `ss` at it: `ss.gdn_state`, and each QSA state's K/V and pooled keys as units
    /// (QsaState::units, device tables on GPU `device` made at the first init and kept).  The drafter's units, when
    /// the pool has them, are for MtpDrafter::use_kv_units.
    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, strata::core::PagePool& pool,
              strata::core::ConversationId id, int device, std::string& err);
    /// Tells whether init has given the conversation its pages.
    bool is_ready() const { return tables_.is_ready(); }
    /// Gives the conversation the pages for appending positions below `tokens` (and the spare indexer block after
    /// them), zeroes the new pages and writes their units' addresses into the QSA states' tables.  Call before each
    /// append; false when the pool has no room.
    bool reserve(int64_t tokens, std::string& err);
    /// Tells whether the existing unit tables already cover the requested append and spare indexer block.
    bool can_reserve_without_growth(int64_t tokens) const;
    /// Points `ss` and the device tables at the pages the pool holds the conversation in now, after a shrink of the
    /// pool moved them (PagePool::resize); the bytes moved with them.
    bool refresh(strata::core::SessionState& ss, std::string& err);
    /// Gives back the growing pages past the first `tokens` tokens (0: a new conversation); the fixed region stays.
    void truncate(int64_t tokens);
    /// Gives back every page, the fixed region's too, once they are copied elsewhere (a paused conversation); `init`
    /// gives the conversation its pages again, keeping the device tables the kernels read.
    void release();
    /// Gets the tokens the pool counts for a conversation whose positions below `tokens` are written: the blocks
    /// `reserve(tokens)` takes, as tokens.
    int64_t get_pool_tokens(int64_t tokens) const;
    /// Gets the conversation's id in its pool.
    strata::core::ConversationId id() const { return tables_.id(); }
    /// Gets the bytes of the pool's pages the conversation holds (0 before init).
    uint64_t bytes() const { return tables_.bytes(); }
    /// Gets the device table of the drafter's units (MtpDrafter::use_kv_units); null when the pool has none.
    const int64_t* get_drafter_units() const { return tables_.table(drafter_region_); }

private:
    /// Gets the QSA layers of the range session `ss` was carved for (its qsa_alloc is 1 for a range without one).
    static int64_t count_qsa_layers(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss);

    /// Its place in the pool and, per QSA layer, the device table of units its state's `units` points at; then the
    /// drafter's (drafter_region_).  An entry is a unit's address less kKvUnitOrigin (kv_unit_table's).
    strata::core::ConversationBlockTables tables_;
    size_t drafter_region_ = 0;                  ///< the pool's region of the drafter's units; 0 (fixed): none
    int64_t idx_block_ = 0;
    int64_t max_blocks_ = 0;                     ///< entries of each table (the session's max_cells in blocks, + 2)
    int64_t reserved_ = 0;                       ///< the tokens the last reserve() gave pages for
};

}  // namespace strata::qwen38
