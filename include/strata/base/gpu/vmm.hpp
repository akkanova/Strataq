#pragma once
// GPU virtual memory (CUDA, and HIP from ROCm 6): an address range reserved once, whose physical memory is mapped and
// unmapped in chunks of a whole number of the device's granularity (2 MiB by default).  The elastic K/V (--kv-grow)
// grows with the context by taking chunks the expert cache gives up, and the paged engines' page pools grow and shrink
// at the top of one range, so that a pool's chunks are back to back (get_gpu_chunk_memory); nothing moves, so every
// captured graph and stored pointer stays valid.  A chunk is one physical chunk, or several mapped back to back
// (map_pieces).  CUDA's driver calls are reached through the runtime's entry-point query, so nothing links the driver
// library.

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace strata::core {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// Looks up the CUDA driver function `name` through the runtime's entry-point query; nullptr when the driver lacks
/// it.  CUDA builds only.
void* find_driver_entry(const char* name);
/// Sets `f` to the driver function `name` (see find_driver_entry); false, `f` unchanged, when there is none.  Used by
/// vmm.cpp and by ExpertCache's segmented arena.
template <class F> bool find_driver_entry(const char* name, F& f) {
    void* p = find_driver_entry(name);
    if (p != nullptr) f = reinterpret_cast<F>(p);
    return p != nullptr;
}

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The default chunk size, a whole number of the device's granularity (0 when not available).
uint64_t vmm_granularity();
/// Gets the smallest size a physical chunk may be a whole number of: the device's granularity (0 when not available).
uint64_t vmm_min_granularity();
/// A new physical chunk of `bytes` (a whole number of vmm_min_granularity; 0: vmm_granularity) on the current device
/// (0: out of memory), and its release.
VmmChunk vmm_chunk_new(uint64_t bytes = 0);
void vmm_chunk_free(VmmChunk h);

class VmmRange {
public:
    VmmRange() = default;
    ~VmmRange() { release(); }
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses on the current device, nothing mapped, in chunks of
    /// `chunk_bytes` (a whole number of vmm_min_granularity; 0: vmm_granularity).
    bool reserve(uint64_t bytes, uint64_t chunk_bytes = 0);
    /// Unmaps and frees every chunk, then the addresses.
    void release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    uint64_t chunk_bytes() const { return chunk_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const { return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0; }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// Readable and writable when it returns; false: out of memory (what was mapped stays mapped).
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Unmaps chunk i and hands back its physical chunk (0 when it was not mapped, or when it was mapped from several
    /// pieces, which then stays mapped: see free_chunk).
    VmmChunk unmap(int64_t i);
    /// Maps chunk i, not mapped yet, from new physical chunks of `piece_bytes` (a whole number of vmm_min_granularity;
    /// the last one takes the rest of the chunk) back to back, and calls `on_piece` with each once it is readable and
    /// writable.  False when out of memory or when `on_piece` returns false; rolls back the mappings.  The paged
    /// engines' GPU pools map a chunk in pieces on an integrated GPU (get_gpu_chunk_memory).
    bool map_pieces(int64_t i, uint64_t piece_bytes,
                    const std::function<bool(uint8_t* piece, uint64_t bytes)>& on_piece = {});
    /// Unmaps chunk i and frees every physical chunk behind it; retains failed unmaps for a later retry.
    void free_chunk(int64_t i);

private:
    /// Keeps ownership during map_pieces, and afterward for multi-piece chunks or incomplete rollbacks.
    struct Pieces {
        uint64_t bytes = 0;            ///< the size of each but the last, which takes the rest of the chunk
        std::vector<VmmChunk> h;       ///< in address order
    };
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Readable and writable [lo, hi) (all newly mapped); when that fails they are unmapped and freed again, so no
    /// chunk counts as mapped without access.
    bool commit_run(int64_t lo, int64_t hi);
    /// Releases consecutive mappings from the start of chunk i; the final piece may be shorter than `piece_bytes`.
    /// Returns false with failed unmaps retained for a later retry.
    bool free_pieces(int64_t i, uint64_t piece_bytes, std::vector<VmmChunk>& pieces);
    // The backend's calls, on `bytes` of the range from byte `offset` (vmm.cpp has one implementation per backend).
    bool map_at(uint64_t offset, uint64_t bytes, VmmChunk h);
    bool set_access_at(uint64_t offset, uint64_t bytes);
    bool unmap_at(uint64_t offset, uint64_t bytes);
    unsigned long long base_ = 0;
    uint64_t chunk_ = 0;        ///< the bytes of each chunk
    int dev_ = 0;               ///< the device the range was reserved on, which gets access to its chunks
    // Nonzero also blocks reuse while a failed rollback still owns mappings.
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk (map_pieces': the first), 0 = unmapped
    std::map<int64_t, Pieces> pieces_;
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (mapped(i)) {
            if (run >= 0 && !commit_run(run, i)) return false;
            run = -1;
            continue;
        }
        VmmChunk h = take();
        if (h == 0) h = vmm_chunk_new(chunk_);
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_free(h);
            if (run >= 0) commit_run(run, i);
            return false;
        }
        if (run < 0) run = i;
    }
    return run < 0 || commit_run(run, hi);
}

}  // namespace strata::core
