// src/common/experts/expert_profile.cpp - reads and writes the expert profile file.  Read the header first.
#include "strata/common/experts/expert_profile.hpp"
#include "strata/base/platform/os.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>

namespace strata::core {

namespace {
/// Bytes of a profile's body: where they start and how many.
using Bytes = std::pair<const void*, size_t>;
bool write_profile_atomically(const std::string& path, const char* who, const uint32_t (&hdr)[5],
                              std::initializer_list<Bytes> body, std::string& err);
}  // namespace

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file, so a later
    // format is refused by its number rather than read as if it were this one
    if (version != 1) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    return write_profile_atomically(path, "write_expert_profile", {1u, (uint32_t) n_layers, (uint32_t) n_expert,
                                    (uint32_t) ranked.size(), (uint32_t) ranked.size()},
                                    {{pairs.data(), pairs.size() * 2}, {table.data(), table.size() * 4}}, err);
}

bool read_expert_counts(const std::string& path, int64_t n_expert, uint32_t max_rows,
                        std::vector<uint32_t>& counts, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto bytes = f.tellg();
    if (!f || bytes < 0) {
        err = "read_expert_counts: cannot open " + path;
        return false;
    }
    f.seekg(0);
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    const bool has_header = f.read(magic, 4) && std::memcmp(magic, "STRP", 4) == 0;
    if (n_expert <= 0 || n_expert > UINT32_MAX ||
        (has_header && (!f.read(reinterpret_cast<char*>(hdr), sizeof hdr) || hdr[0] != 2 || hdr[2] != n_expert))) {
        err = "read_expert_counts: " + path + " is not a version 2 profile of " + std::to_string(n_expert) +
              " experts a row (version " + std::to_string(hdr[0]) + ", " + std::to_string(hdr[2]) + " experts)";
        return false;
    }
    const uint64_t row = 4 * (uint64_t) n_expert, body = (uint64_t) bytes - (has_header ? 24 : 0);
    const uint64_t n_rows = has_header ? hdr[1] : body / row;
    if (n_rows > max_rows || body / row > max_rows ||
        (body / row == max_rows && body % row != 0) ||
        n_rows > counts.max_size() / (uint64_t) n_expert) {
        err = "read_expert_counts: " + path + " has too many rows for this model";
        return false;
    }
    if (n_rows > body / row) {
        err = "read_expert_counts: the counts in " + path + " are truncated";
        return false;
    }
    std::vector<uint32_t> loaded((size_t) n_rows * (size_t) n_expert);
    f.clear();
    f.seekg(has_header ? 24 : 0);
    if (!f.read(reinterpret_cast<char*>(loaded.data()), (std::streamsize) (loaded.size() * 4))) {
        err = "read_expert_counts: the counts in " + path + " are truncated";
        return false;
    }
    counts = std::move(loaded);
    return true;
}

bool write_expert_counts(const std::string& path, int64_t n_rows, int64_t n_expert,
                         const std::vector<uint32_t>& counts, std::string& err) {
    if (n_rows <= 0 || n_rows > UINT32_MAX || n_expert <= 0 || n_expert > UINT32_MAX ||
        counts.size() > SIZE_MAX / 4 || counts.size() != (uint64_t) n_rows * (uint64_t) n_expert) {
        err = "write_expert_counts: the counts are not n_rows x n_expert";
        return false;
    }
    return write_profile_atomically(path, "write_expert_counts", {2u, (uint32_t) n_rows, (uint32_t) n_expert, 0u, 0u},
                                    {{counts.data(), counts.size() * 4}}, err);
}

namespace {

/// Publishes a complete profile by replacing the destination. `who` names the caller in errors.
bool write_profile_atomically(const std::string& path, const char* who, const uint32_t (&hdr)[5],
                              std::initializer_list<Bytes> body, std::string& err) {
    // Version 1 retains its temporary path and diagnostics; counts may be saved by separate engines.
    static std::atomic<uint64_t> next_save{0};
    const std::string tmp = path + ".tmp" + (hdr[0] == 2 ? "." +
        std::to_string(platform::get_process_id()) + "." + std::to_string(next_save.fetch_add(1)) : "");
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = std::string(who) + ": cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5;
    for (const auto& [data, n] : body)
        ok = ok && std::fwrite(data, 1, n, f) == n;
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok)
        std::filesystem::rename(tmp, path, ec);   // a Windows reader may deny replacement; keep the old file
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = std::string(who) + ": cannot write " + path;
        return false;
    }
    return true;
}

}  // namespace

}  // namespace strata::core
