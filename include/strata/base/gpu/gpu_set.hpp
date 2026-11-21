#pragma once
// What a set of GPUs has in common, which picks an engine's defaults.  Used by Qwen3.8's apply_device_defaults and
// the DeepSeek engine.

#include "strata/base/gpu/gfx_arch.hpp"

#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace strata::core {

/// Traits that every GPU of a set has; all false for an empty set or when the runtime cannot describe one of them.
struct GpuSetFacts {
    bool all_integrated = false;   ///< each uses the PC's RAM as its memory (Strix Halo, DGX Spark, an APU)
    bool same_arch = false;        ///< (HIP builds only) the first seven gcnArchName characters match, ignoring flags
    bool all_gfx12 = false;        ///< (HIP builds only) each is RDNA4 (gfx1200 / gfx1201)
};

/// Asks the GPU runtime for the GpuSetFacts of the GPUs numbered `devices`.
inline GpuSetFacts read_gpu_set_facts(const std::vector<int>& devices) {
    GpuSetFacts f;
    f.all_integrated = !devices.empty();
#if defined(STRATA_USE_HIP)
    f.same_arch = f.all_gfx12 = !devices.empty();
    cudaDeviceProp first{};
#endif
    for (size_t i = 0; i < devices.size(); ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, devices[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            return GpuSetFacts{};
        }
        f.all_integrated = f.all_integrated && p.integrated != 0;
#if defined(STRATA_USE_HIP)
        if (i == 0)
            first = p;
        f.same_arch = f.same_arch && std::strncmp(p.gcnArchName, first.gcnArchName, 7) == 0;
        f.all_gfx12 = f.all_gfx12 && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
#endif
    }
    return f;
}

}  // namespace strata::core
