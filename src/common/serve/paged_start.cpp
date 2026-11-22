// src/common/serve/paged_start.cpp - what every model's start does before the paged engine runs it (paged_engine.hpp,
// "a model's start"): the engine's own options, the runtime's environment, the cards' links, the pools' sizes.
#include "strata/common/serve/paged_engine.hpp"

#include "strata/base/gpu/links.hpp"
#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/gpu/vmm.hpp"
#include "strata/base/platform/env.hpp"
#include "strata/base/platform/memory_manager.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>

namespace strata::serve {

/// The bytes each way that measure a GPU's link to RAM at start: the old engine's probe's burst (probe_pcie_h2d_gbps
/// in src/models/qwen38/model.cpp), about 10 ms on PCIe 4.0 x16 and 90 ms on 3.0 x4.
constexpr size_t kLinkProbeBytes = 256ull << 20;

namespace {
strata::core::LinkSpeed get_host_link(int dev);
}  // namespace

int take_paged_options(int& argc, char** argv, const std::string& program, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
        std::string* text = nullptr;   ///< instead of `value`, for a word
    };
    int64_t rows = s.step.rows, lookahead = -1, concurrent = -1, mtp_rows = 0;   // -1: chosen after parsing
    std::string drafter = "on", save_on_shutdown = "off", gpu_mib, popularity_decay, draft_rule;
    const Own own[] = {{"--tokens-min", &s.tokens_min},       {"--tokens-max", &s.tokens_max},
                       {"--conversation-tokens", &s.conversation_context},
                       {"--vram-conversations", &s.vram_conversations},
                       {"--conversation-idle-seconds", &s.conversation_idle_seconds},
                       {"--decode-share-percent", &s.decode_share_percent},
                       {"--prompt-turn-ms", &s.prompt_turn_ms},
                       {"--gpu-mib", nullptr, &gpu_mib},      {"--ram-mib", &s.ram_mib},
                       {"--ssd-pause-mib", &s.ssd_pause_mib}, {"--rows", &rows},
                       {"--mtp-max-t", &mtp_rows},            {"--lookup-chain", &s.lookup_chain},
                       {"--lookup-chain-min", &s.lookup_chain_min},
                       {"--suffix-draft", &s.suffix_draft},    {"--lookahead", &lookahead},
                       {"--concurrent", &concurrent}, {"--batch-groups", nullptr, &s.batch_groups},
                       {"--pause-dir", nullptr, &s.pause_dir}, {"--copy-back", nullptr, &s.copy_back},
                       {"--drafter", nullptr, &drafter},      {"--remote-experts", nullptr, &s.remote_experts},
                       {"--draft-rule", nullptr, &draft_rule},
                       {"--save-on-shutdown", nullptr, &save_on_shutdown},
                       {"--prompt-order", nullptr, &s.prompt_order},
                       {"--expert-popularity-decay", nullptr, &popularity_decay},
                       {"--step-profile", &s.step_profile}};
    const char* const name = program.c_str();
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const Own* o = std::find_if(std::begin(own), std::end(own),
                                    [&](const Own& x) { return std::strcmp(argv[i], x.name) == 0; });
        if (o == std::end(own)) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s: %s needs a value\n", name, o->name);
            return 2;
        }
        if (o->text != nullptr)
            *o->text = argv[++i];
        else
            *o->value = std::atoll(argv[++i]);
    }
    argc = out;
    if (s.step_profile != 0 && s.step_profile != 1) {
        std::fprintf(stderr, "%s: --step-profile takes 0 or 1\n", name);
        return 2;
    }
    s.step.rows = (int) std::clamp<int64_t>(rows, 0, INT32_MAX);
    s.step.lookahead = (int) std::clamp<int64_t>(lookahead, -1, INT32_MAX);
    s.step.concurrent = (int) std::clamp<int64_t>(concurrent, -1, INT32_MAX);   // -1: --batch, set once it is read
    s.drafter = drafter == "on";
    s.save_on_shutdown = save_on_shutdown == "on";
    s.step.model_lookahead = mtp_rows > 0 ? (int) std::min<int64_t>(mtp_rows - 1, 7) : -1;
    s.suffix_draft = std::clamp<int64_t>(s.suffix_draft, 0, INT32_MAX);
    s.decode_share_percent = std::clamp<int64_t>(s.decode_share_percent, 0, INT32_MAX);
    s.prompt_turn_ms = std::clamp<int64_t>(s.prompt_turn_ms, 0, INT32_MAX);
    s.lookup_chain = std::clamp<int64_t>(s.lookup_chain, 0, 7);
    s.lookup_chain_min = std::clamp<int64_t>(s.lookup_chain_min, 3, INT32_MAX);
    if (s.batch_groups != "auto" && (s.batch_groups.empty() ||
        s.batch_groups.find_first_not_of("0123456789") != std::string::npos ||
        std::strtoull(s.batch_groups.c_str(), nullptr, 10) < 1 ||
        std::strtoull(s.batch_groups.c_str(), nullptr, 10) > INT32_MAX)) {
        std::fprintf(stderr, "%s: --batch-groups takes auto or a positive integer\n", name);
        return 2;
    }
    if (!popularity_decay.empty() &&
        !((s.popularity_decay = std::strtod(popularity_decay.c_str(), nullptr)) > 0 && s.popularity_decay <= 1)) {
        std::fprintf(stderr, "%s: --expert-popularity-decay takes a factor above 0 and at most 1\n", name);
        return 2;
    }
    for (const auto& [flag, word, words] : {std::tuple{"--copy-back", &s.copy_back, "auto|on|off"},
                                            {"--remote-experts", &s.remote_experts, "auto|on|off"},
                                            {"--prompt-order", &s.prompt_order, "auto|layer|chunk"},
                                            {"--drafter", &drafter, "on|off"},
                                            {"--save-on-shutdown", &save_on_shutdown, "on|off"}})
        if (word->find('|') != std::string::npos ||
            ("|" + std::string(words) + "|").find("|" + *word + "|") == std::string::npos) {
            std::fprintf(stderr, "%s: %s takes %s\n", name, flag, words);
            return 2;
        }
    if (strata::kernels::DraftRule rule; !draft_rule.empty()) {
        if (!strata::kernels::parse_draft_rule(draft_rule, rule)) {
            std::fprintf(stderr, "%s: --draft-rule takes %s\n", name, strata::kernels::kDraftRuleChoices);
            return 2;
        }
        s.draft_rule = rule;
    }
    if (std::string err; !gpu_mib.empty() && !parse_i64_list(gpu_mib.c_str(), s.gpu_mib, err)) {
        std::fprintf(stderr, "%s: --gpu-mib takes MiB, one per card of a layer split: %s\n", name, err.c_str());
        return 2;
    }
    return -1;
}

void set_paged_runtime_env(const PagedSettings& s, bool hops_possible) {
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // ROCclr on Windows keeps freed memory in its resource cache and gives none of it back (on the Strix Halo, 8 GiB
    // freed with hipHostFree or hipFree stayed taken from Windows' RAM), so the pools could not shrink
    strata::platform::set_env("GPU_RESOURCE_CACHE_SIZE", "0", false);
#endif
    // A hop needs a free queue beside the owner's spinning verify window. Extra queues are requested only for
    // separate GPUs: same-GPU stages cannot hop (get_hop_refusal). Whether separate NVIDIA cards need the extra
    // connections remains for the NVIDIA session's explicit-default-queues check to measure.
    if (s.remote_experts != "off" && hops_possible) {
        strata::platform::set_env("GPU_MAX_HW_QUEUES", "16", false);
        strata::platform::set_env("CUDA_DEVICE_MAX_CONNECTIONS", "32", false);
    }
    // Each model reads its switch at its first prompt (STRATA_LM_HIP: Qwen3.8's opt-in on AMD); one the user set wins.
    if (s.prompt_order != "auto")
        for (const char* name : {"STRATA_PREFILL_LAYER_MAJOR", "DS4_LAYER_MAJOR", "STRATA_LM_HIP"})
            strata::platform::set_env(name, s.prompt_order == "layer" ? "1" : "0", false);
}

PagedCard& add_paged_card(PagedCards& cards, int dev, int64_t lb, int64_t le, int64_t n_layers, bool trim_weights,
                          const std::string& program, uint64_t expert_bytes) {
    PagedCard& card = cards.emplace_back();
    card.index = cards.size() - 1;
    card.dev = dev;
    card.lb = lb;
    card.le = le;
    card.weights_lb = trim_weights ? lb : 0;
    card.weights_le = trim_weights ? le : n_layers;
    const bool shared = std::any_of(cards.begin(), cards.end() - 1, [&](const PagedCard& c) { return c.dev == dev; });
    card.id = "gpu" + std::to_string(dev) + (shared ? "." + std::to_string(card.index) : "");
    const strata::core::LinkSpeed link = card.link = get_host_link(dev);
    card.refill_batch = strata::core::get_refill_batch(link, expert_bytes, kRefillBudgetMs, kRefillBatch);
    std::fprintf(stderr, "%s: %s's link to RAM: %.1f GB/s to the GPU, %.1f GB/s back, %.3f ms a round trip\n",
                 program.c_str(), card.id.c_str(), link.to_gbps, link.from_gbps, link.latency_ms);
    card.layer_ms = strata::program::split_rules::get_layer_ms(strata::core::get_card_speed(dev));
    return card;
}

CardPoolPlan plan_card_pool(const PagedCard& card, const PagedSettings& s, const strata::core::StateLayout& layout,
                            int64_t max_context, uint64_t page_bytes, const std::vector<uint64_t>& blob_bytes,
                            const std::vector<strata::core::ExpertKey>& ranking, int64_t max_experts, uint64_t keep) {
    uint64_t most = get_conversation_bytes(layout, max_context, page_bytes);
    for (size_t i = 0; i < ranking.size() && (int64_t) i < max_experts; ++i)
        most += (blob_bytes[(size_t) ranking[i].layer] + page_bytes - 1) / page_bytes * page_bytes;
    CardPoolPlan plan;
    plan.gpu_mib = s.get_gpu_mib(card.index);
    plan.room = get_gpu_room(card.dev, keep);
    plan.bytes = plan.gpu_mib >= 0 ? std::min((uint64_t) plan.gpu_mib << 20, plan.room) : std::min(plan.room, most);
    plan.config.page_bytes = page_bytes;
    plan.config.chunk_bytes = get_chunk_bytes(plan.bytes, page_bytes);
    plan.config.tokens_min = std::clamp<int64_t>(s.tokens_min, 0, max_context);
    return plan;
}

std::vector<int64_t> plan_start_split(const std::vector<int>& devices, const PagedSettings& s, uint64_t keep,
                                      const strata::core::LayerSizes& sizes, std::vector<int32_t> ranked_layers,
                                      const std::string& program) {
    std::vector<uint64_t> pools;
    std::vector<double> layer_ms, links;
    for (size_t i = 0; i < devices.size(); ++i) {
        const int dev = devices[i];
        const int64_t mib = s.get_gpu_mib(i);
        pools.push_back(mib < 0 ? get_gpu_room(dev, keep) / std::count(devices.begin(), devices.end(), dev) :
                                  (uint64_t) mib << 20);
        layer_ms.push_back(strata::program::split_rules::get_layer_ms(strata::core::get_card_speed(dev)));
        links.push_back(get_host_link(dev).to_gbps);
    }
    strata::core::SplitProblem problem =
        strata::core::make_paged_split_problem(sizes, pools, layer_ms, links, 1, s.tokens_min);
    problem.ranked_layers = std::move(ranked_layers);
    const std::string prefix = program + ": layer split auto: ";
    const strata::core::SplitPlan plan = strata::core::plan_layer_split(problem, prefix.c_str());
    strata::core::print_layer_split(plan, problem.ranked_layers.size(), prefix.c_str());
    return plan.cuts;
}

strata::core::ChunkMemory get_gpu_chunk_memory(int dev, uint64_t chunk_bytes, bool zeroed) {
    strata::core::ChunkMemory memory;
    memory.copy = [dev](uint8_t* to, const uint8_t* from, uint64_t n) {
        const strata::core::OnDevice on(dev);
        return strata::core::gpu_copy(to, from, (size_t) n);
    };
    const strata::core::OnDevice on(dev);
    std::string name;
    uint64_t total = 0, free = 0;
    bool unified = false;
    strata::core::read_gpu_memory(dev, name, total, free, unified);
    auto range = std::make_shared<strata::core::VmmRange>();
    const uint64_t granularity = strata::core::vmm_min_granularity();
    if (granularity > 0 && chunk_bytes % granularity == 0 && range->reserve(total, chunk_bytes)) {
        memory.contiguous = true;
        memory.alloc = [dev, zeroed, range](uint64_t n) -> uint8_t* {   // the lowest chunk of the range not mapped
            const strata::core::OnDevice on(dev);
            int64_t i = 0;
            while (i < range->chunks() && range->mapped(i))
                ++i;
            if (n != range->chunk_bytes() || i == range->chunks() ||
                !range->map_range(i, i + 1, [] { return (strata::core::VmmChunk) 0; }))
                return nullptr;
            uint8_t* p = range->base() + (uint64_t) i * n;
            if (zeroed && !(strata::core::gpu_memset_async(p, 0, (size_t) n, nullptr) &&
                            strata::core::gpu_stream_synchronize(nullptr))) {
                strata::core::vmm_chunk_free(range->unmap(i));
                return nullptr;
            }
            strata::core::get_memory_manager().add(p, n, "gpu" + std::to_string(dev), {});
            return p;
        };
        memory.free = [dev, range](uint8_t* p) {
            const strata::core::OnDevice on(dev);
            strata::core::get_memory_manager().remove(p);
            const auto i = (int64_t) ((uint64_t) (p - range->base()) / range->chunk_bytes());
            strata::core::vmm_chunk_free(range->unmap(i));
        };
        return memory;
    }
    memory.alloc = [dev, zeroed](uint64_t n) {
        const strata::core::OnDevice on(dev);
        // Adapters initialize pages on use; touching free runs can page active weights out under WDDM.
        return (uint8_t*) (zeroed ? strata::core::gpu_alloc_zeroed((size_t) n) :
                                   strata::core::gpu_alloc((size_t) n));
    };
    memory.free = [dev](uint8_t* p) {
        const strata::core::OnDevice on(dev);
        strata::core::gpu_free(p);
    };
    return memory;
}

uint64_t get_conversation_bytes(const strata::core::StateLayout& layout, int64_t tokens, uint64_t page_bytes) {
    uint64_t bytes = layout.bytes_for(tokens);
    for (const strata::core::StateRegion& r : layout.regions)
        bytes += strata::core::choose_block_pages(r, page_bytes) * page_bytes;
    return bytes;
}

uint64_t get_chunk_bytes(uint64_t pool_bytes, uint64_t page_bytes) {
    const uint64_t granularity = strata::core::vmm_min_granularity();
    const uint64_t unit = granularity > 0 ? std::lcm(page_bytes, granularity) : page_bytes;
    return std::max<uint64_t>(std::min(kChunkBytes, pool_bytes) / unit, 1) * unit;
}

void say_gpu_pool(const std::string& program, const PagedCard& card, const CardPoolPlan& plan) {
    const strata::core::PagePool& pool = card.experts.pool();
    char capped[128] = "";
    if (plan.gpu_mib >= 0 && ((uint64_t) plan.gpu_mib << 20) > plan.room)
        std::snprintf(capped, sizeof capped, "; --gpu-mib %lld asked for more than the GPU has room for",
                      (long long) plan.gpu_mib);
    std::fprintf(stderr, "%s: the page pool of %s (layers %lld-%lld): %.2f GiB (of %.2f GiB it could take) in chunks "
                         "of %.0f MiB%s, pages of %llu bytes, for the experts and the conversations, %lld tokens of it "
                         "kept for the conversations%s\n",
                 program.c_str(), card.id.c_str(), (long long) card.lb, (long long) card.le - 1,
                 (double) pool.report().total.bytes / 1073741824.0, (double) plan.room / 1073741824.0,
                 (double) pool.chunk_bytes() / 1048576.0,
                 card.experts.is_contiguous() ? " mapped back to back in one virtual range" : "",
                 (unsigned long long) pool.page_bytes(),
                 (long long) pool.tokens_min(), capped);
}

uint64_t get_ram_pool_bytes(const PagedSettings& s, uint64_t headroom, uint64_t most) {
    return s.ram_mib < 0 ? std::min(get_ram_room(headroom), most) : (uint64_t) s.ram_mib << 20;
}

strata::core::PagePoolConfig get_ram_pool_config(const PagedCard& card) {
    strata::core::PagePoolConfig config;
    config.page_bytes = card.experts.pool().page_bytes();
    config.chunk_bytes = kChunkBytes / config.page_bytes * config.page_bytes;
    return config;
}

std::vector<strata::core::ExpertKey> get_ram_ranking(const std::vector<strata::core::ExpertKey>& profile,
                                                     const std::vector<uint64_t>& blob_bytes, int64_t n_expert) {
    const auto at = [&](int64_t l, int64_t e) { return (size_t) (l * n_expert + e); };
    std::vector<strata::core::ExpertKey> ranking = profile;
    std::vector<uint8_t> ranked(blob_bytes.size() * (size_t) n_expert, 0);
    for (const strata::core::ExpertKey& k : profile)
        ranked[at(k.layer, k.expert)] = 1;
    for (int32_t l = 0; l < (int32_t) blob_bytes.size(); ++l)
        for (int32_t e = 0; e < (int32_t) n_expert; ++e)
            if (!ranked[at(l, e)])
                ranking.push_back({l, e});
    return ranking;
}

bool open_ram_pool(const std::string& program, const PagedSettings& s, strata::core::ExpertSource& files,
                   uint64_t headroom, int ring, const strata::core::StateLayout& layout, const PagedCard& card,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert,
                   const std::vector<strata::core::ExpertKey>& profile, const uint8_t* gpu_bytes,
                   uint64_t prompt_bytes, strata::core::RamExperts& ram, double& copy_back_gain, std::string& err) {
    uint64_t every = 0;
    for (const uint64_t b : blob_bytes)
        every += b * (uint64_t) n_expert;
    const uint64_t bytes = get_ram_pool_bytes(
        s, headroom, every + (uint64_t) ring * *std::max_element(blob_bytes.begin(), blob_bytes.end()));
    const auto open_t0 = std::chrono::steady_clock::now();
    strata::core::PagePoolConfig config = get_ram_pool_config(card);
    // (a lent run stays inside one chunk: the RAM's chunks are separate allocations)
    config.chunk_bytes = std::max(config.chunk_bytes, (prompt_bytes + config.page_bytes - 1) / config.page_bytes *
                                                          config.page_bytes);
    if (!ram.open(files, bytes, ring, layout, config, blob_bytes, n_expert,
                  get_ram_ranking(profile, blob_bytes, n_expert), err))
        return false;
    const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - open_t0).count();
    double ssd = 0, back = 0;
    // (no more than `gpu_bytes`' chunk holds: a small GPU pool's chunk is smaller than the staging ring)
    const uint64_t timed = std::min(ram.ring_bytes(), card.experts.pool().chunk_bytes());
    // the best of three: on the Strix Halo single copies measured 3.3 to 57 GB/s, single SSD reads 0.3 to 2.4 GB/s
    for (int round = 0; round < 3; ++round) {
        ssd = std::max(ssd, ram.time_ssd_reads(round));
        back = std::max(back, ram.time_copy_in(gpu_bytes, timed));
    }
    copy_back_gain = s.copy_back == "on"    ? HUGE_VAL
                     : s.copy_back == "off" ? 0.0
                                            : strata::core::get_copy_back_gain(back, ssd);
    std::fprintf(stderr, "%s: the RAM's page pool: %.2f GiB%s, opened in %.1f s; it takes experts while the engine "
                         "serves, and an expert it does not hold is read from the SSD without the file cache.  An "
                         "expert the GPU evicts is %s (copying it back: %.1f GB/s, reading the SSD: %.1f GB/s; "
                         "--copy-back %s)\n",
                 program.c_str(), (double) (ram.pages().pool().report().total.bytes) / 1073741824.0,
                 ram.is_page_locked() ? ", page-locked" : "", open_s,
                 strata::core::is_copy_back_worth(copy_back_gain, false) ? "copied back to RAM"
                 : strata::core::is_copy_back_worth(copy_back_gain, true)
                     ? "copied back to RAM when a prompt's buffers evict it, else dropped"
                     : "dropped",
                 back, ssd, s.copy_back.c_str());
    return true;
}

namespace {

/// Reuses each GPU's first successful probe for the start split and all its cards.
strata::core::LinkSpeed get_host_link(int dev) {
    static std::map<int, strata::core::LinkSpeed> measured;
    strata::core::LinkSpeed& link = measured[dev];
    if (!link.is_known())
        link = strata::core::measure_host_link(dev, kLinkProbeBytes);
    return link;
}

}  // namespace

}  // namespace strata::serve
