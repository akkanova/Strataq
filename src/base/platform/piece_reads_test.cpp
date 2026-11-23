// src/base/platform/piece_reads_test.cpp - read_pieces against the bytes the test wrote: pieces at sector edges,
// overlapping, merged and not, in two files, at a file's end and past it.  No GPU, no model files.
#include "strata/base/platform/piece_reads.hpp"
#include "strata/base/platform/memory_manager.hpp"
#include "strata/base/platform/os.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace strata::platform;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// Gets the byte test file `file` holds at `at`.
uint8_t get_byte(int file, uint64_t at) {
    return (uint8_t) ((at * 2654435761u >> 13) + (uint64_t) file * 77);
}

}  // namespace

int main() {
#if defined(_WIN32) || defined(__linux__)
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "piece_reads_test";
    std::filesystem::create_directories(dir);
    const uint64_t sizes[2] = {40ull << 20, (2ull << 20) + 1234};   // over one 32 MiB request; ends mid-sector
    fd_t fds[2];
    for (int f = 0; f < 2; ++f) {
        const std::string path = (dir / (f == 0 ? "a.bin" : "b.bin")).string();
        std::vector<uint8_t> data(sizes[f]);
        for (uint64_t i = 0; i < sizes[f]; ++i)
            data[i] = get_byte(f, i);
        std::FILE* file = std::fopen(path.c_str(), "wb");
        check(file != nullptr && std::fwrite(data.data(), 1, data.size(), file) == data.size(), "a file is written");
        std::fclose(file);
        fds[f] = open_direct(path);
    }
    if (fds[0] == kBadFd || fds[1] == kBadFd) {   // (tmpfs refuses O_DIRECT)
        for (fd_t f : fds)
            close_fd(f);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::printf("piece_reads_test: skipped, the temporary directory has no unbuffered reads\n");
        return 0;
    }
    struct Range { int file; uint64_t off, bytes; };
    const std::vector<Range> ranges = {
        {0, 0, 4096}, {0, 4095, 2}, {0, 4096, 1}, {0, 8191, 1},   // a sector, across and at its edges
        {0, 5000, 70000},                                         // overlapping the pieces above
        {0, 300001, 50003},                                       // under 1 MiB on (merged on Windows only)
        {0, 3u << 20, 777}, {0, (3ull << 20) + 1000, 33ull << 20},   // over 1 MiB on; together over 32 MiB
        {0, sizes[0] - 10, 10}, {1, 7, 31}, {1, 4090, 12}, {1, sizes[1] - 1234, 1234},   // file ends; file b
        {0, 4095, 2},                                             // a range twice
    };
    std::vector<std::vector<uint8_t>> got(ranges.size());
    std::vector<FilePiece> pieces;
    for (size_t k = 0; k < ranges.size(); ++k) {
        got[k].assign(ranges[k].bytes + 1, 0xA5);   // (+1: dst unaligned, and a neighbour byte to check)
        pieces.push_back({fds[ranges[k].file], ranges[k].off, ranges[k].bytes, got[k].data() + k % 2});
    }
    for (int pass = 0; pass < 2; ++pass) {   // (the second reuses this thread's staging buffer)
        check(read_pieces(pieces.data(), pieces.size()), "the batch reads");
        for (size_t k = 0; k < ranges.size(); ++k) {
            bool same = got[k][k % 2 == 0 ? ranges[k].bytes : 0] == 0xA5;
            for (uint64_t i = 0; i < ranges[k].bytes; ++i)
                same = same && got[k][i + k % 2] == get_byte(ranges[k].file, ranges[k].off + i);
            check(same, "each piece holds its range's bytes and nothing beside it");
        }
    }
    uint8_t past[16];
    const FilePiece past_end = {fds[1], sizes[1] - 8, sizeof past, past};
    check(!read_pieces(&past_end, 1), "a piece past the end of its file fails");
    const FilePiece zero = {kBadFd, 17, 0, nullptr};
    check(read_pieces(&zero, 1), "a zero-length piece needs no file or destination");
    const FilePiece overflow = {fds[0], UINT64_MAX - 1, 16, past};
    check(!read_pieces(&overflow, 1), "a range whose end wraps is refused");
    const FilePiece huge = {fds[0], 0, (1ull << 63) - 4096, past};
    const FilePiece huge_batch[] = {huge, huge, huge};
    check(!read_pieces(huge_batch, 3), "a staging size whose sum wraps is refused");
#if defined(_WIN32)
    const FilePiece too_large = {fds[0], 0, 1ull << 32, past};
    check(!read_pieces(&too_large, 1), "a read larger than DWORD is refused before allocation");
#endif
    const FilePiece bad_batch[] = {pieces[0], {std::numeric_limits<fd_t>::max(), 0, 16, past}};
    check(!read_pieces(bad_batch, 2), "a failed submission drains any earlier reads");
    check(read_pieces(pieces.data(), pieces.size()), "a batch succeeds after failed reads");
    check(read_pieces(nullptr, 0), "an empty batch reads");
    const auto before = strata::core::get_memory_manager().get_blocks("ram");
    std::vector<std::thread> readers;
    int results[4] = {};
    for (size_t t = 0; t < 4; ++t)
        readers.emplace_back([&, t] {
            std::vector<uint8_t> data(70000);
            const FilePiece piece = {fds[t % 2], 5000, data.size(), data.data()};
            for (int pass = 0; pass < 8; ++pass) {
                if (!read_pieces(&piece, 1))
                    return;
                for (size_t i = 0; i < data.size(); ++i)
                    if (data[i] != get_byte((int) (t % 2), piece.off + i))
                        return;
            }
            results[t] = 1;
        });
    for (std::thread& reader : readers)
        reader.join();
    for (int result : results)
        check(result == 1, "threads read the same handles concurrently");
    const auto after = strata::core::get_memory_manager().get_blocks("ram");
    check(before.size() == after.size() && before.size() == 1 && before[0].bytes == after[0].bytes,
          "reader threads release their staging allocations on exit");
    for (fd_t f : fds)
        close_fd(f);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
#endif
    std::printf("piece_reads_test: %d checks passed\n", checks);
    return 0;
}
