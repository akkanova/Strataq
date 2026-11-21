// src/models/qwen38/conversation_pages.cpp - a Qwen3.8 conversation's state in a GPU page pool
// (conversation_pages.hpp).
#include "strata/models/qwen38/conversation_pages.hpp"
#include "strata/models/qwen38/kernels/qsa.hpp"

#include "strata/base/gpu/runtime.hpp"
#include "strata/core/layout.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/models/qwen38/layer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace strata::qwen38 {

bool ConversationPages::get_layout(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss,
                                   const strata::core::QsaState* drafter, strata::core::StateLayout& layout,
                                   std::string& err) {
    using namespace strata::core;
    const int64_t n_qsa = g.n_qsa_layers(), n_gdn = g.n_gdn_layers(), qsa_here = count_qsa_layers(g, ss);
    if (qsa_here == 0 || ss.gdn_alloc == 0 || ss.gdn_state != nullptr) {
        err = "the page pool needs a session carved without its state (session_set_pool_state) for a range of layers "
              "with a GDN and a QSA layer";
        return false;
    }
    const strata::kernels::QsaShapes s = strata::kernels::qsa_shapes(g);
    const auto fmt = (strata::kernels::KvFormat) qsa_kv_format(ss.qsa_states[ss.qsa_primary()]);
    try {
        const strata::kernels::KvUnitLayout unit = strata::kernels::kv_unit_layout(s, fmt);
        const uint64_t key_bytes = (uint64_t) s.idx_dim * sizeof(float);
        if (!describe_conversation(g, (unit.bytes - key_bytes) / (uint64_t) s.idx_block, layout, err))
            return false;
        layout.regions.erase(layout.regions.begin());   // the small running state stays in the session
        if (layout.regions.size() != (size_t) (1 + n_qsa) || layout.regions[1].unit_bytes != unit.bytes) {
            err = "the page pool's layout does not match the K/V's units (" + std::to_string(unit.bytes) + " B)";
            return false;
        }
        layout.regions[0].unit_bytes = layout.regions[0].unit_bytes / (uint64_t) n_gdn * (uint64_t) ss.gdn_alloc;
        layout.regions.resize((size_t) (1 + qsa_here));   // the session's own layers
        if (drafter != nullptr)
            layout.add(RegionKind::per_block,
                       strata::kernels::kv_unit_layout(s, (strata::kernels::KvFormat) qsa_kv_format(*drafter)).bytes, 1,
                       s.idx_block, "Drafter context");
    } catch (const std::exception& e) {
        err = std::string("the page pool: ") + e.what();
        return false;
    }
    return true;
}

bool ConversationPages::init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss,
                             strata::core::PagePool& pool, strata::core::ConversationId id, int device,
                             std::string& err) {
    using namespace strata::core;
    const int64_t n_qsa = count_qsa_layers(g, ss);
    const std::vector<StateRegion>& regions = pool.layout().regions;
    size_t fixed_regions = 0;
    while (fixed_regions < regions.size() && regions[fixed_regions].kind == RegionKind::fixed)
        ++fixed_regions;
    drafter_region_ = (int64_t) (regions.size() - fixed_regions) == n_qsa + 1 ? regions.size() - 1 : 0;
    idx_block_ = strata::kernels::qsa_shapes(g).idx_block;
    max_blocks_ = ss.max_cells / idx_block_ + 2;
    std::vector<ConversationBlockTables::TableSize> sizes(regions.size());
    for (size_t r = fixed_regions; r < regions.size(); ++r)
        sizes[r] = {max_blocks_, pool.block_bytes(r) / regions[r].unit_bytes};
    // QSA reads live units and the spare at n_bid; reserve() publishes those before a kernel can use them.
    if (!tables_.attach(pool, id, device, sizes, strata::kernels::kKvUnitOrigin, err))
        return false;
    pool.set_conversation_label(id, id == 1 ? "Main conversation" : "Conversation " + std::to_string(id - 1));
    ss.gdn_state = (float*) tables_.get_fixed_address(0);
    for (int64_t j = 0; j < n_qsa; ++j)
        qsa_use_units(g, ss.qsa_states[ss.qsa_ord0 + j], tables_.table(fixed_regions + (size_t) j));
    if (!gpu_memset_async(ss.gdn_state, 0, pool.block_bytes(0), nullptr) || !gpu_stream_synchronize(nullptr)) {
        err = "the page pool: zeroing the conversation's recurrent state failed";
        tables_.release();
        return false;
    }
    return true;
}

bool ConversationPages::reserve(int64_t tokens, std::string& err) {
    const int64_t need = get_pool_tokens(tokens);
    if (need <= reserved_)
        return true;
    if (!tables_.grow(need, err))
        return false;
    reserved_ = need;
    return true;
}

bool ConversationPages::refresh(strata::core::SessionState& ss, std::string& err) {
    ss.gdn_state = (float*) tables_.get_fixed_address(0);
    return tables_.refresh(err);
}

void ConversationPages::truncate(int64_t tokens) {
    reserved_ = tokens > 0 ? std::min(((tokens - 1) / idx_block_ + 2) * idx_block_, reserved_) : 0;
    tables_.truncate(reserved_);
}

void ConversationPages::release() {
    tables_.release();
    reserved_ = 0;
}

int64_t ConversationPages::get_pool_tokens(int64_t tokens) const {
    // blocks up to the last position's, and the spare one after it that the indexer writes ahead
    return std::min((std::max<int64_t>(tokens, 1) - 1) / idx_block_ + 2, max_blocks_) * idx_block_;
}

int64_t ConversationPages::count_qsa_layers(const strata::core::ModelGeometry& g,
                                            const strata::core::SessionState& ss) {
    const int64_t interval = std::max<int64_t>(g.qsa_interval, 1);
    return g.n_qsa_layers() > 0 ? ss.layer_hi / interval - ss.layer_lo / interval : 0;
}

bool ConversationPages::can_reserve_without_growth(int64_t tokens) const {
    return is_ready() && tables_.has_blocks_for(get_pool_tokens(tokens));
}

}  // namespace strata::qwen38
