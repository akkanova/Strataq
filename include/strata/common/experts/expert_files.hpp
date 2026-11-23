// include/strata/common/experts/expert_files.hpp - the routed experts read from a model's files without a file
// mapping, for the new engines' RAM page pools (RamExperts), which read through it the experts they do not hold.
// strata-ds41 describes its GGUF or EXL3 pack; strata-q38 gets its layout from FileExpertSource once its load is done.
#pragma once

#include "strata/base/platform/os.hpp"
#include "strata/core/expert_source.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// Where one layer's routed experts lie in the model's files (ExpertFiles::open's `paths`) and where each matrix goes
/// in an expert's blob, the layout the GPU's kernels and the CPU lane read: matrix r (gate, up, down) of expert e is
/// bytes[r] bytes at off[r] + e * stride[r] in file file[r], at tr[r] in a blob of `blob` bytes.  A matrix of 0 bytes
/// is not read (experts.bin keeps a whole blob as matrix 0).
struct ExpertFileLayout {
    int n_exp = 0;
    int file[3] = {-1, -1, -1};
    uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};
    uint64_t bytes[3] = {0, 0, 0};
    uint64_t tr[3] = {0, 0, 0};
    uint64_t blob = 0;
    /// EXL3 stores nine separately placed pieces per expert, in the pack's file order.
    struct Piece {
        int file = -1;
        uint64_t off = 0, bytes = 0, dst = 0;
    };
    std::vector<std::vector<Piece>> pieces;
};

/// The routed experts of a pack as an ExpertSource that copies each into the blob layout the model reads
/// (ExpertFileLayout), a batch's pieces read as one queue (platform::read_pieces).  It maps nothing: whether it reads
/// around the file cache is settled when it opens.  Copies only: blob() has no memory to give.
class ExpertFiles final : public ExpertSource {
public:
    ExpertFiles() = default;
    ~ExpertFiles() override;
    ExpertFiles(const ExpertFiles&) = delete;
    ExpertFiles& operator=(const ExpertFiles&) = delete;

    /// Opens `paths` for the experts of `layers` (each layer's file numbers index `paths`): for unbuffered reads when
    /// a test read of every file succeeds that way, else through the file cache (is_direct).  `inputs` becomes
    /// model_inputs().  False (and `err`) when a file can be read neither way.
    bool open(const std::vector<std::string>& paths, std::vector<ExpertFileLayout> layers, std::string& err,
              std::vector<std::pair<std::string, std::string>> inputs = {});
    /// Tells whether the reads go around the file cache.
    bool is_direct() const { return is_direct_; }
    /// Gets the experts copy_blobs read, their bytes and the ms the reads took, for --stats.
    int64_t reads() const override { return reads_.load(std::memory_order_relaxed); }
    uint64_t get_read_bytes() const { return read_bytes_.load(std::memory_order_relaxed); }
    double get_read_ms() const { return (double) read_us_.load(std::memory_order_relaxed) / 1000.0; }

    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// Safe from several threads.
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;

private:
    bool open_files(const std::vector<std::string>& paths, bool direct, std::string& err);
    void close_files();

    std::vector<platform::fd_t> files_;   ///< per file, opened as is_direct_ says
    std::vector<ExpertFileLayout> layers_;
    bool is_direct_ = false;
    std::atomic<int64_t> reads_{0};
    std::atomic<uint64_t> read_bytes_{0}, read_us_{0};
};

}  // namespace strata::core
