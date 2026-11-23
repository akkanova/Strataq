// src/common/experts/expert_files.cpp - routed experts read without a file mapping (expert_files.hpp).
#include "strata/common/experts/expert_files.hpp"

#include "strata/base/platform/piece_reads.hpp"

#include <chrono>

namespace strata::core {

ExpertFiles::~ExpertFiles() {
    close_files();
}

bool ExpertFiles::open(const std::vector<std::string>& paths, std::vector<ExpertFileLayout> layers, std::string& err,
                       std::vector<std::pair<std::string, std::string>> inputs) {
    layers_ = std::move(layers);
    inputs_ = std::move(inputs);
    is_direct_ = false;
    for (const bool direct : {true, false}) {
        if (open_files(paths, direct, err)) {
            is_direct_ = direct;
            err.clear();
            return true;
        }
    }
    return false;
}

bool ExpertFiles::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const int32_t l = (int32_t) layer, e = (int32_t) expert;
    return copy_blobs(&l, &e, &dst, 1);
}

bool ExpertFiles::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    thread_local std::vector<platform::FilePiece> pieces;   // (kept for this thread's next batch)
    pieces.clear();
    uint64_t bytes = 0;
    for (size_t i = 0; i < n; ++i) {
        const ExpertFileLayout& l = layers_[(size_t) layers[i]];
        const size_t count = l.pieces.empty() ? 3 : l.pieces[(size_t) experts[i]].size();
        for (size_t r = 0; r < count; ++r) {
            const auto piece = l.pieces.empty()
                ? ExpertFileLayout::Piece{l.file[r], l.off[r] + (uint64_t) experts[i] * l.stride[r],
                                          l.bytes[r], l.tr[r]}
                : l.pieces[(size_t) experts[i]][r];
            if (piece.bytes == 0)
                continue;
            pieces.push_back({(intptr_t) files_[(size_t) piece.file], piece.off, piece.bytes, dst[i] + piece.dst});
            bytes += piece.bytes;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = platform::read_pieces(pieces.data(), pieces.size());
    reads_.fetch_add((int64_t) n, std::memory_order_relaxed);
    read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    read_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
    return ok;
}

/// Opens every file of `paths` unbuffered (`direct`) or through the file cache, and reads its first byte to prove
/// that read_pieces can read it that way; false (nothing left open, `err` naming the file) when one fails.
bool ExpertFiles::open_files(const std::vector<std::string>& paths, bool direct, std::string& err) {
    close_files();
    for (const std::string& path : paths) {
        files_.push_back(direct ? platform::open_direct(path) : platform::open_par(path));
        uint8_t first = 0;
        const platform::FilePiece probe{(intptr_t) files_.back(), 0, 1, &first};
        if (files_.back() == platform::kBadFd || !platform::read_pieces(&probe, 1)) {
            err = path + (files_.back() == platform::kBadFd ? ": cannot open" : ": cannot read");
            close_files();
            return false;
        }
    }
    return true;
}

/// Closes the files open_files opened.
void ExpertFiles::close_files() {
    for (const platform::fd_t f : files_)
        if (f != platform::kBadFd)
            platform::close_fd(f);
    files_.clear();
}

}  // namespace strata::core
