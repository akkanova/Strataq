// Checks the pack-to-arena byte layout with synthetic files, including a staged-round boundary.
#include "strata/core/weights.hpp"
#include "strata/common/artifact/tensor_directory.hpp"
#include "strata/common/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using strata::core::WeightTable;
namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

/// Removes the synthetic pack even when a load fails.
struct TempDir {
    fs::path path = fs::temp_directory_path() / fs::u8path("strata-weights-\xc3\xa9-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    TempDir tmp;
    const auto utf8 = tmp.path.u8string();
    const std::string dir(utf8.begin(), utf8.end());
    const uint64_t count = strata::artifact::kStageRound / 4 + 32;
    std::vector<uint32_t> bf(count);
    std::vector<uint16_t> high(count), halves(65536);
    std::vector<float> wide(halves.size()), f32(halves.size());
    for (size_t i = 0; i < count; ++i) {
        high[i] = (uint16_t) (0x3e00 + i % 256);
        bf[i] = (uint32_t) high[i] << 16;
    }
    for (size_t i = 0; i < halves.size(); ++i) {
        halves[i] = (uint16_t) i;
        wide[i] = strata::kernels::f32_from_f16(halves[i]);
        f32[i] = (float) ((int) i - 32768) / 128;
    }
    const uint64_t bf_bytes = count * 2, f16_bytes = f32.size() * 2, scale_bytes = wide.size() * 4;
    const uint64_t pool = bf_bytes + f16_bytes + 4 + scale_bytes + 4;
    auto write_index = [&](bool odd = false) {
        std::ofstream ix(tmp.path / "index.txt");
        ix << "# align 256 pool " << pool << " tensors 4\n"
           << "skip 2 0 0 0 0 0 32 1 8 0 32 0 0 0 0 0 0 0\n"
           << "bf 3 1 16 " << count * 4 << " 0 " << bf_bytes << ' ' << count
           << " 1 0 0 0 0 0 0 0 0 0 0\n"
           << "half 0 3 0 " << f32.size() * 4 << ' ' << bf_bytes << ' ' << f16_bytes << ' ' << f32.size()
           << " 1 0 0 0 0 0 0 0 0 0 0\n"
           << "planes 1 0 0 " << 8 + halves.size() * 2 - odd << ' ' << bf_bytes + f16_bytes << ' '
           << 8 + scale_bytes - 2 * odd << " 32 1 4 -8 32 0 1 4 " << scale_bytes - 2 * odd << " 4 1 0\n";
    };
    write_index();
    {
        std::ofstream out(tmp.path / "extra.bin", std::ios::binary);
        out.write("0123456789abcdef", 16);
        out.write((const char*) bf.data(), bf.size() * 4);
        std::ofstream(tmp.path / "dense.bin", std::ios::binary).write((const char*) f32.data(), f32.size() * 4);
        std::ofstream planes(tmp.path / "embd.bin", std::ios::binary);
        planes.write("code", 4);
        planes.write((const char*) halves.data(), halves.size() * 2);
        planes.write("tail", 4);
    }
    const std::set<std::string> skip{"skip"};
    std::string err;
    uint64_t needed = 0;
    check(WeightTable::pool_bytes(dir, needed, err, &skip) && needed >= pool, "compacted pool: " + err);
    void* arena = nullptr;
    if (cudaMalloc(&arena, needed) != cudaSuccess)
        return 1;
    WeightTable table;
    check(table.load(dir, arena, needed, err, &skip), "load UTF-8 path, extra.bin and three transforms: " + err);
    auto compare = [&](const char* name, const void* expected, size_t bytes) {
        const auto* w = table.find(name);
        std::vector<char> got(bytes);
        check(w && w->bytes == bytes &&
              cudaMemcpy(got.data(), w->data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
              std::memcmp(got.data(), expected, bytes) == 0, std::string(name) + " bytes match");
    };
    compare("bf", high.data(), high.size() * 2);
    std::vector<uint16_t> expected(f32.size());
    for (size_t i = 0; i < f32.size(); ++i)
        expected[i] = strata::kernels::f16_from_f32(f32[i]);
    compare("half", expected.data(), expected.size() * 2);
    std::vector<char> planes(8 + scale_bytes);
    std::memcpy(planes.data(), "code", 4);
    std::memcpy(planes.data() + 4, wide.data(), scale_bytes);
    std::memcpy(planes.data() + 4 + scale_bytes, "tail", 4);
    compare("planes", planes.data(), planes.size());
    const auto* skipped = table.find("skip");
    check(skipped && !skipped->resident && !skipped->data, "shape-only row needs no experts.bin");
    std::vector<uint16_t> host;
    check(table.read_bf16_host("bf", host, err) && host == high, "host bf16 read spans two rounds");
    fs::resize_file(tmp.path / "extra.bin", 17);
    check(!table.read_bf16_host("bf", host, err), "short host read fails");
    check(!table.load(dir, arena, needed, err, &skip), "short upload read fails");
    const std::set<std::string> skip_bf{"skip", "bf"};
    write_index(true);
    check(!table.load(dir, arena, needed, err, &skip_bf) && err.find("odd") != std::string::npos,
          "odd fp16 plane is refused before conversion");
    std::ofstream(tmp.path / "index.txt", std::ios::app) << "broken row\n";
    int bits = -1;
    check(!WeightTable::pool_bytes(dir, needed, err) && !WeightTable::index_code_bits(dir, "bf", bits, err),
          "both index queries reject a malformed row");
    check(cudaFree(arena) == cudaSuccess, "release arena");
    return failures ? 1 : 0;
}
