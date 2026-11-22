// GGUF and EXL3 routed experts read without the file cache
// (expert_files.hpp).
#include "expert_files.hpp"

#include "strata/base/platform/piece_reads.hpp"

#include <algorithm>

namespace strata::ds4 {

ExpertFiles::~ExpertFiles() {
    for (const platform::fd_t f : direct_)
        platform::close_fd(f);
    for (const platform::fd_t f : buffered_)
        platform::close_fd(f);
}

bool ExpertFiles::open(const std::vector<std::string>& paths, std::vector<ExpertFileLayout> layers, std::string& err) {
    layers_ = std::move(layers);
    for (const std::string& path : paths) {
        direct_.push_back(platform::open_direct(path));
        buffered_.push_back(platform::open_par(path));
        if (buffered_.back() == platform::kBadFd) {
            err = path + ": cannot open";
            return false;
        }
    }
    is_direct_ = std::none_of(direct_.begin(), direct_.end(), [](platform::fd_t f) { return f == platform::kBadFd; });
    return true;
}

bool ExpertFiles::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const int32_t l = (int32_t) layer, e = (int32_t) expert;
    return copy_blobs(&l, &e, &dst, 1);
}

bool ExpertFiles::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    thread_local std::vector<platform::FilePiece> pieces;   // (kept for this thread's next batch)
    thread_local std::vector<int> files;                     // pieces[k]'s file
    pieces.clear();
    files.clear();
    for (size_t i = 0; i < n; ++i) {
        const ExpertFileLayout& l = layers_[(size_t) layers[i]];
        const size_t count = l.pieces.empty() ? 3 : l.pieces[(size_t) experts[i]].size();
        for (size_t r = 0; r < count; ++r) {
            const auto piece = l.pieces.empty()
                ? ExpertFileLayout::Piece{l.file[r], l.off[r] + (uint64_t) experts[i] * l.stride[r],
                                          l.bytes[r], l.tr[r]}
                : l.pieces[(size_t) experts[i]][r];
            pieces.push_back({direct_[(size_t) piece.file], piece.off, piece.bytes, dst[i] + piece.dst});
            files.push_back(piece.file);
        }
    }
    if (is_direct_ && platform::read_pieces(pieces.data(), pieces.size()))
        return true;
    for (size_t k = 0; k < pieces.size(); ++k)
        if (!platform::pread_par(buffered_[(size_t) files[k]], pieces[k].dst, pieces[k].bytes, pieces[k].off))
            return false;
    return true;
}

}  // namespace strata::ds4
