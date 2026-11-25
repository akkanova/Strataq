// metal/tests/runtime_hello.cpp - the Metal runtime end to end, with the shared graph cache built unchanged:
//   1. gpu::malloc_device / host_alloc (mapped) / memcpy_async / a kernel through metal::dispatch / events / D2H
//   2. the SHARED include/strata/base/gpu/shape_graph_cache.hpp, compiled unchanged for Metal: capture a 3-kernel
//      "window" once, replay it 5 times, check the data path is live (each replay reads the buffers' current
//      contents, as a CUDA graph does)
//   3. host_device_pointer: a kernel writes a mapped host buffer that the host reads without a copy
//   4. the cost of replaying a 1,000-dispatch window
#include "strata/base/gpu/shape_graph_cache.hpp"
#include "strata/gpu/metal.hpp"
#include "strata/gpu/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gpu = strata::gpu;

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Axpy { device const float* x; device float* y; float a; uint n; };
kernel void axpy(constant Axpy& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) p.y[i] = p.a * p.x[i] + p.y[i];
}
struct Scale { device float* y; device const float* s; uint n; };   // the factor is DATA (a device buffer), not an argument
kernel void scale(constant Scale& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) p.y[i] *= p.s[0];
}
)MSL";
struct Axpy { uint64_t x, y; float a; uint32_t n; };
struct Scale { uint64_t y, s; uint32_t n; };

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    std::string err;
    MTL::ComputePipelineState* p_axpy = gpu::metal::pipeline("hello", kSrc, "axpy", err);
    MTL::ComputePipelineState* p_scale = gpu::metal::pipeline("hello", kSrc, "scale", err);
    CHECK(p_axpy && p_scale, "pipelines: %s", err.c_str());
    if (!p_axpy || !p_scale) return 1;
    const uint32_t N = 1 << 16;
    const gpu::metal::Grid groups{(N + 255) / 256, 1, 1}, threads{256, 1, 1};

    gpu::Stream s = nullptr, s2 = nullptr;
    CHECK(gpu::stream_create(&s, gpu::kStreamNonBlocking) == gpu::kSuccess, "stream");
    CHECK(gpu::stream_create(&s2, gpu::kStreamNonBlocking) == gpu::kSuccess, "stream 2");
    float *dx = nullptr, *dy = nullptr, *ds = nullptr, *hx = nullptr;
    CHECK(gpu::malloc_device((void**) &dx, N * 4) == gpu::kSuccess, "malloc x");
    CHECK(gpu::malloc_device((void**) &dy, N * 4) == gpu::kSuccess, "malloc y");
    CHECK(gpu::malloc_device((void**) &ds, 4) == gpu::kSuccess, "malloc s");
    CHECK(gpu::host_alloc((void**) &hx, N * 4, gpu::kHostAllocMapped) == gpu::kSuccess, "host_alloc");
    CHECK(gpu::host_view(dx) != nullptr && gpu::host_view(dx) != (void*) dx, "a device pointer is a GPU address, not the CPU view");

    // ---- 1. copies, a kernel, an event across two streams, D2H
    std::vector<float> x(N), y(N, 1.0f), got(N);
    for (uint32_t i = 0; i < N; ++i) x[i] = (float) (i % 1000);
    std::memcpy(hx, x.data(), N * 4);
    gpu::memcpy_async(dx, hx, N * 4, gpu::kHostToDevice, s);   // mapped host buffer -> device: a blit
    gpu::memcpy_async(dy, y.data(), N * 4, gpu::kHostToDevice, s);   // pageable -> device
    Axpy a{(uint64_t) dx, (uint64_t) dy, 2.0f, N};
    gpu::metal::dispatch(s, p_axpy, &a, sizeof a, groups, threads);
    gpu::Event ev = nullptr;
    gpu::event_create(&ev, gpu::kEventDisableTiming);
    gpu::event_record(ev, s);
    gpu::stream_wait_event(s2, ev);   // s2's kernel runs after s's
    Axpy a2{(uint64_t) dx, (uint64_t) dy, 1.0f, N};
    gpu::metal::dispatch(s2, p_axpy, &a2, sizeof a2, groups, threads);
    gpu::memcpy_async(got.data(), dy, N * 4, gpu::kDeviceToHost, s2);
    gpu::stream_synchronize(s2);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) bad += got[i] != 3.0f * x[i] + 1.0f;
    CHECK(bad == 0, "step 1: %d wrong values", bad);
    std::printf("1. copies + kernel + cross-stream event + D2H: %s\n", bad ? "FAIL" : "OK");

    // ---- 2. the shared ShapeGraphCache
    {
        float one = 1.0f;
        gpu::memcpy_sync(ds, &one, 4, gpu::kHostToDevice);
        strata::core::ShapeGraphCache<int> graphs;
        Axpy w{(uint64_t) dx, (uint64_t) dy, 1.0f, N};
        Scale sc{(uint64_t) dy, (uint64_t) ds, N};
        int bodies = 0;
        const auto window = [&] {
            ++bodies;
            gpu::metal::dispatch(s, p_axpy, &w, sizeof w, groups, threads);
            gpu::metal::dispatch(s, p_scale, &sc, sizeof sc, groups, threads);
            gpu::metal::dispatch(s, p_axpy, &w, sizeof w, groups, threads);
        };
        std::vector<float> m(got);   // host model of y
        for (int rep = 0; rep < 5; ++rep) {
            const float f = rep % 2 ? 0.5f : 2.0f;   // the per-step value changes as DATA between replays
            float* sv = (float*) gpu::host_view(ds);
            gpu::stream_synchronize(s);
            *sv = f;
            CHECK(graphs.run(0, s, window, err), "run: %s", err.c_str());
            for (uint32_t i = 0; i < N; ++i) m[i] = ((m[i] + x[i]) * f) + x[i];
        }
        gpu::stream_synchronize(s);
        gpu::memcpy_sync(got.data(), dy, N * 4, gpu::kDeviceToHost);
        int wrong = 0;
        for (uint32_t i = 0; i < N; ++i) wrong += got[i] != m[i];
        CHECK(wrong == 0 && bodies == 1, "graph replays: %d wrong values, %d captures", wrong, bodies);
        std::printf("2. shared ShapeGraphCache: 3-kernel window captured once, 5 replays with live data: %s\n",
                    wrong == 0 && bodies == 1 ? "OK" : "FAIL");
    }

    // ---- 3. a mapped host buffer written by a kernel, read by the host with no copy
    {
        float* dview = nullptr;
        CHECK(gpu::host_device_pointer((void**) &dview, hx) == gpu::kSuccess, "host_device_pointer");
        std::memset(hx, 0, N * 4);
        Axpy w{(uint64_t) dx, (uint64_t) dview, 1.0f, N};
        gpu::metal::dispatch(s, p_axpy, &w, sizeof w, groups, threads);
        gpu::stream_synchronize(s);
        int wrong = 0;
        for (uint32_t i = 0; i < N; ++i) wrong += hx[i] != x[i];
        CHECK(wrong == 0, "mapped: %d wrong", wrong);
        std::printf("3. kernel -> mapped host buffer, host reads it in place: %s\n", wrong ? "FAIL" : "OK");
    }

    // ---- 4. replay cost of a 1,000-dispatch window
    {
        strata::core::ShapeGraphCache<int> graphs;
        Axpy w{(uint64_t) dx, (uint64_t) dy, 0.0f, 256};
        const auto window = [&] {
            for (int k = 0; k < 1000; ++k)
                gpu::metal::dispatch(s, p_axpy, &w, sizeof w, gpu::metal::Grid{1, 1, 1}, threads);
            return true;
        };
        const cudaGraphExec_t g = graphs.capture(0, s, window, "capture: ", err);
        for (int rep = 0; rep < 5; ++rep) {
            const double t0 = now_ms();
            cudaGraphLaunch(g, s);
            const double t1 = now_ms();
            gpu::stream_synchronize(s);
            std::printf("4. 1000-dispatch window: replay (encode+commit) %.3f ms, until complete %.3f ms\n", t1 - t0, now_ms() - t0);
        }
    }
    const auto st = gpu::metal::stats();
    std::printf("stats: %llu dispatches, %llu command buffers, %llu blits, %llu host syncs\n", (unsigned long long) st.dispatches,
                (unsigned long long) st.command_buffers, (unsigned long long) st.blits, (unsigned long long) st.host_syncs);
    gpu::free_device(dx); gpu::free_device(dy); gpu::free_device(ds); gpu::free_host(hx);
    gpu::event_destroy(ev);
    gpu::stream_destroy(s); gpu::stream_destroy(s2);
    std::printf(g_fail ? "RESULT: %d FAILED\n" : "RESULT: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
