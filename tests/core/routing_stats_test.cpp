// Checks the GPU's routing counts (routing_stats.hpp): rows that share an expert, passes and layers counted apart,
// and the switch read when a recorded graph runs (Qwen3.8 replays its decode windows).
#include "strata/common/kernels/routing_stats.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>

int main() {
    using namespace strata::core;
    using namespace strata::kernels;
    // two rows of 3 picks, 4 ids apart: experts 31, 32, 511 and 32, 511, 400 (4 distinct, 5 with the shared one)
    const std::array<int32_t, 8> ids = {31, 32, 511, -1, 32, 511, 400, -1};
    auto* device = static_cast<int32_t*>(gpu_alloc(sizeof(ids)));
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    init_routing_counts(2, 512, 1);
    const auto expect = [&](const char* what, int64_t routings, int64_t experts) {
        const bool synced = gpu_stream_synchronize(stream);
        const RoutingCounts c = get_routing_counts();
        if (synced && c.routings == routings && c.experts == experts)
            return true;
        std::fprintf(stderr, "routing_stats_test: %s: expected %lld/%lld, got %lld/%lld\n", what, (long long) routings,
                     (long long) experts, (long long) c.routings, (long long) c.experts);
        return false;
    };
    const bool ok = device && gpu_copy(device, ids.data(), sizeof(ids)) &&
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
        (count_routing(device, 2, 3, 4, 0, stream), expect("off", 0, 0)) &&
        (set_routing_counting(true), count_routing(device, 2, 3, 4, 0, stream), expect("one pass", 8, 5)) &&
        (count_routing(device, 2, 3, 4, 0, stream), count_routing(device, 2, 3, 4, 1, stream),
         expect("another pass and layer", 24, 15)) &&
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess &&
        (count_routing(device, 2, 3, 4, 0, stream), cudaStreamEndCapture(stream, &graph) == cudaSuccess) &&
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess &&
        (set_routing_counting(false), cudaGraphLaunch(exec, stream) == cudaSuccess) && expect("graph, off", 24, 15) &&
        (set_routing_counting(true), cudaGraphLaunch(exec, stream) == cudaSuccess) && expect("graph, on", 32, 20);
    if (ok)
        std::puts("routing_stats_test: passed");
    return ok ? 0 : 1;
}
