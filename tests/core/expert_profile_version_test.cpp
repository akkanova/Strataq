// read_expert_profile refuses a profile whose header version is not 1, naming the version it found, the one it
// reads, and the file - before the geometry check, since the fields after the version are only known to be a
// version 1 layout.  A version 1 file still loads.  DeepSeek's counts (read_expert_counts, version 2) round-trip,
// refuse another version, expert count or a truncated file, and still read a headerless file from before version 2.
// CPU only: no device is touched.
#include "strata/common/experts/expert_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
// overwrite the little-endian uint32 `version` field at byte 4 (after "STRP") in place
bool set_version(const std::string& path, uint32_t v) {
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    if (f == nullptr) return false;
    uint8_t le[4];
    for (int b = 0; b < 4; ++b) le[b] = (uint8_t) (v >> (8 * b));
    const bool ok = std::fseek(f, 4, SEEK_SET) == 0 && std::fwrite(le, 1, 4, f) == 4;
    return (std::fclose(f) == 0) && ok;
}
bool has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }
}  // namespace

int main() {
    using Pair = std::pair<int32_t, int32_t>;
    const int64_t L = 2, E = 3;
    const std::vector<Pair> ranked = {{1, 2}, {0, 0}, {1, 0}};
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_profile_version_test";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "profile.bin").string();
    std::string err;
    std::vector<Pair> back;
    int64_t slots = 0;

    // version 1, as --expert-profile-save and tools/make_profile.py write it: loads
    check(strata::core::write_expert_profile(path, L, E, ranked, err), "written");
    check(strata::core::read_expert_profile(path, L, E, back, slots, err) && back == ranked && slots == 3,
          "a version 1 profile loads");

    // This fixture is the version 1 wire format, including its reverse-rank table.
    const std::vector<uint32_t> expected_v1 = {0x50525453, 1, 2, 3, 3, 3,
                                             0x00020001, 0, 1, 1, UINT32_MAX, UINT32_MAX,
                                             2, UINT32_MAX, 0};
    std::ifstream v1(path, std::ios::binary);
    const std::string v1_bytes((std::istreambuf_iterator<char>(v1)), std::istreambuf_iterator<char>());
    check(v1_bytes == std::string(reinterpret_cast<const char*>(expected_v1.data()), expected_v1.size() * 4),
          "version 1 bytes stay unchanged");
    v1.close();

    // the same bytes with the version field at 2: refused, naming 2, 1 and the file
    check(set_version(path, 2), "version field rewritten to 2");
    err.clear();
    check(!strata::core::read_expert_profile(path, L, E, back, slots, err), "a version 2 profile is refused");
    check(has(err, "read_expert_profile: ") && has(err, path), "the refusal names the file");
    check(has(err, "version 2") && has(err, "version 1"), "the refusal names the version found and the one read");

    // version 0 (a zeroed header) is not version 1 either
    check(set_version(path, 0), "version field rewritten to 0");
    check(!strata::core::read_expert_profile(path, L, E, back, slots, err) && has(err, "version 0"),
          "a version 0 profile is refused by its number");

    // the version is checked before the geometry: a version 2 file for another model is refused as version 2
    check(set_version(path, 2), "version field rewritten to 2 again");
    check(!strata::core::read_expert_profile(path, L + 1, E, back, slots, err) && has(err, "version 2"),
          "a version 2 profile of another geometry is refused by its version, not its geometry");

    // and back to 1: it loads again, so the refusal was the version field and nothing else
    check(set_version(path, 1), "version field rewritten to 1");
    check(strata::core::read_expert_profile(path, L, E, back, slots, err) && back == ranked, "version 1 loads again");

    // version 2: DeepSeek's routing counts, n_rows x n_expert
    const int64_t rows = 3;
    std::vector<uint32_t> counts((size_t) (rows * E)), got;
    for (size_t i = 0; i < counts.size(); ++i)
        counts[i] = (uint32_t) (i * 7 + 1);
    const std::string cpath = (dir / "counts.bin").string();
    using strata::core::read_expert_counts;
    check(strata::core::write_expert_counts(cpath, rows, E, counts, err), "counts written");
    check(std::filesystem::file_size(cpath) == 24 + counts.size() * 4, "a counts file is the header and the counts");
    check(read_expert_counts(cpath, E, rows, got, err) && got == counts, "a version 2 counts file round-trips");
    check(!read_expert_counts(cpath, E + 1, rows, got, err), "counts of another number of experts a row are refused");
    check(!read_expert_counts(path, E, rows, got, err) && has(err, "version 1"), "a version 1 profile is not counts");
    check(!strata::core::write_expert_counts(cpath, rows, E + 1, counts, err), "counts of the wrong size are refused");
    check(!read_expert_counts(cpath, E, rows - 1, got, err) && has(err, "too many rows"),
          "the model row limit is checked before allocating");
    check(!strata::core::read_expert_profile(cpath, rows, E, back, slots, err) && has(err, "version 2"),
          "version 2 counts cannot load as version 1 rankings");
    check(set_version(cpath, 7), "counts version rewritten");
    check(!read_expert_counts(cpath, E, rows, got, err) && has(err, "version 7"),
          "an unknown counts version cannot load as legacy counts");
    check(set_version(cpath, 2), "counts version restored");
    std::filesystem::resize_file(cpath, 24 + counts.size() * 4 - 1);
    check(!read_expert_counts(cpath, E, rows, got, err) && has(err, "truncated"), "truncated counts are refused");
    std::filesystem::resize_file(cpath, 8);
    check(!read_expert_counts(cpath, E, rows, got, err), "a truncated STRP header cannot load as legacy counts");
    std::FILE* f = std::fopen(cpath.c_str(), "wb");   // headerless, as before version 2, with a partial row after
    bool wrote_legacy = false;
    if (f != nullptr) {
        wrote_legacy = std::fwrite(counts.data(), 4, counts.size(), f) == counts.size() && std::fputc(9, f) == 9;
        wrote_legacy = (std::fclose(f) == 0) && wrote_legacy;
    }
    check(wrote_legacy, "headerless counts written");
    check(read_expert_counts(cpath, E, rows + 1, got, err) && got == counts,
          "a headerless counts file reads its whole rows");
    check(!read_expert_counts(cpath, E, rows - 1, got, err) && has(err, "too many rows"),
          "legacy counts also respect the allocation limit");
    check(!strata::core::write_expert_counts(cpath, INT64_MAX, INT64_MAX, counts, err),
          "writer dimensions cannot overflow or truncate");
    check(strata::core::write_expert_counts(cpath, rows, E, counts, err), "legacy counts upgrade to STRP");

#if defined(_WIN32)
    f = std::fopen(cpath.c_str(), "rb");
    check(f != nullptr, "another engine holds the profile open");
    if (f != nullptr) {
        check(!strata::core::write_expert_counts(cpath, rows, E, counts, err) && has(err, "cannot write"),
              "a Windows reader denying deletion prevents replacement safely");
        std::fclose(f);
        check(read_expert_counts(cpath, E, rows, got, err) && got == counts,
              "failed replacement preserves the existing profile");
    }
#endif
    std::array<std::thread, 4> writers;
    std::array<bool, 4> saved{};
    for (size_t i = 0; i < writers.size(); ++i) {
        writers[i] = std::thread([&, i] {
            std::string write_err;
            const std::vector<uint32_t> own(counts.size(), (uint32_t) i);
            for (int j = 0; j < 16; ++j) {
                if (strata::core::write_expert_counts(cpath, rows, E, own, write_err))
                    saved[i] = true;
            }
        });
    }
    for (auto& writer : writers)
        writer.join();
    check(saved[0] && saved[1] && saved[2] && saved[3], "each concurrent writer can publish");
    check(read_expert_counts(cpath, E, rows, got, err) && got.size() == counts.size() &&
          got.front() < writers.size() && got == std::vector<uint32_t>(counts.size(), got.front()),
          "concurrent saves publish one complete writer's profile");
    check(std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator{}) == 2,
          "saves leave no temporary files");

    std::filesystem::remove_all(dir);
    if (fails == 0) std::puts("expert_profile_version_test: OK");
    return fails == 0 ? 0 : 1;
}
