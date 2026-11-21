// tests/core/shape_graph_cache_test.cu - ShapeGraphCache: one capture per shape, replays with live data, and which
// graph the least-recently-used rule frees, pinned graphs kept (GPU, synthetic).
#include "strata/base/gpu/shape_graph_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}
static void gpu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
__global__ void add(unsigned* value, unsigned n) { *value += n; }

int main() {
    cudaStream_t stream;
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned* value;
    gpu(cudaMalloc(&value, sizeof(unsigned)));
    gpu(cudaMemsetAsync(value, 0, sizeof(unsigned), stream));
    strata::core::ShapeGraphCache<int> cache;
    std::string err;
    int bodies = 0;
    for (int round = 0; round < 3; ++round)
        for (int shape = 1; shape <= 3; ++shape)
            check(cache.run(shape, stream, [&] { ++bodies; add<<<1, 1, 0, stream>>>(value, (unsigned) shape); }, err),
                  err.c_str());
    gpu(cudaStreamSynchronize(stream));
    unsigned got = 0;
    gpu(cudaMemcpy(&got, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(bodies == 3 && cache.size() == 3, "each shape is captured once");
    check(got == 3 * (1 + 2 + 3), "every run replays its shape's graph");
    const cudaGraphExec_t first = cache.find(1);
    check(cache.pinning([&] {
        return cache.capture(1, stream, [] { return false; }, "capture: ", err) == first;
    }), "capturing an existing pinned key keeps its executable");

    const auto evict = [&] { return cache.evict_oldest(stream); };
    check(cache.find(1) && evict() && !cache.find(2) && cache.size() == 2, "the least recently used graph goes first");
    check(cache.pinning([&] { return cache.find(3) && cache.find(1) && !evict(); }), "pinned graphs stay");
    check(evict() && evict() && !evict() && cache.size() == 0, "unpinned after pinning");

    check(!cache.capture(4, stream, [&] {
        add<<<1, 1, 0, stream>>>(value, 100);
        err = "record failed";
        return false;
    }, "capture: ", err) && err == "record failed", "failed recording preserves its error");
    try {
        cache.capture(4, stream, [&]() -> bool {
            add<<<1, 1, 0, stream>>>(value, 100);
            throw std::runtime_error("record failed");
        }, "capture: ", err);
        check(false, "recording exception propagates");
    } catch (const std::runtime_error&) {}
    check(cache.run(4, stream, [&] { add<<<1, 1, 0, stream>>>(value, 4); }, err), err.c_str());
    check(cache.pinning([&] {
        check(cache.find(4) != nullptr, "pin a replay in flight");
        check(cache.pinning([&] { return !evict(); }), "nested pinning keeps outer graphs");
        return !evict();
    }), "nested pinning restores outer pins");
    try {
        cache.pinning([&]() -> bool {
            cache.find(4);
            throw std::runtime_error("pin failed");
        });
        check(false, "pinning exception propagates");
    } catch (const std::runtime_error&) {}
    check(evict(), "exception releases pins and eviction waits for a replay");
    gpu(cudaMemcpy(&got, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(got == 3 * (1 + 2 + 3) + 4, "failed captures do not launch work");
    gpu(cudaFree(value));
    gpu(cudaStreamDestroy(stream));
    std::printf("shape_graph_cache_test: OK\n");
    return 0;
}
