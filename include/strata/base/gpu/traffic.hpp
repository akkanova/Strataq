// strata/base/gpu/traffic.hpp - the bytes moved between host memory and each GPU: copies both ways, and a GPU's reads
// and writes of host memory in place (mapped, page-locked memory), and how many experts went to it.  Each total only
// grows.  The paged engine sends them once a second on its STATS line (contracts/requests.md); the server and the
// web Monitor turn them into rates.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace strata::core {

/// One device's traffic since the program started.
struct GpuTraffic {
    uint64_t to_gpu = 0;     ///< bytes that went from host memory to the device
    uint64_t from_gpu = 0;   ///< bytes that went from the device to host memory
    uint64_t experts = 0;    ///< experts among `to_gpu`; one split over GPUs counts on its first part's
};

/// The device numbers whose traffic is counted; a higher one's is not.
inline constexpr int kMaxCountedDevices = 64;

/// Gets the process-wide totals (to_gpu, from_gpu, experts) by the runtime's device number: process-wide, because the
/// copies happen deep in each model's code, on several threads.  Used by count_gpu_traffic and get_gpu_traffic.
inline std::array<std::array<std::atomic<uint64_t>, 3>, kMaxCountedDevices>& get_gpu_traffic_totals() {
    static std::array<std::array<std::atomic<uint64_t>, 3>, kMaxCountedDevices> totals{};
    return totals;
}

/// Adds bytes that moved between host memory and device `device` (the runtime's number) to its totals, `experts` of
/// them being whole experts sent to it; thread-safe.
inline void count_gpu_traffic(int device, uint64_t to_gpu, uint64_t from_gpu, uint64_t experts = 0) {
    if (device < 0 || device >= kMaxCountedDevices)
        return;
    std::array<std::atomic<uint64_t>, 3>& t = get_gpu_traffic_totals()[(size_t) device];
    t[0].fetch_add(to_gpu, std::memory_order_relaxed);
    t[1].fetch_add(from_gpu, std::memory_order_relaxed);
    if (experts > 0)
        t[2].fetch_add(experts, std::memory_order_relaxed);
}

/// Gets device `device`'s traffic so far; zero for a device that is not counted.
inline GpuTraffic get_gpu_traffic(int device) {
    if (device < 0 || device >= kMaxCountedDevices)
        return {};
    const std::array<std::atomic<uint64_t>, 3>& t = get_gpu_traffic_totals()[(size_t) device];
    return {t[0].load(std::memory_order_relaxed), t[1].load(std::memory_order_relaxed),
            t[2].load(std::memory_order_relaxed)};
}

}  // namespace strata::core
