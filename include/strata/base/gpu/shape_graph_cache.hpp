// include/strata/base/gpu/shape_graph_cache.hpp - one cache of captured GPU graphs per window shape.
//
// An engine replays the same launches for every window of one shape (its row count, its slot layout, the session it
// runs for), so it captures them once per shape and replays the graph. A replay re-reads the buffers but not the
// kernel arguments, which are fixed at capture: whatever changes between windows must be data in a device buffer.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

namespace strata::core {

/// Keeps captured graphs (`cudaGraphExec_t`) by window shape `Key` (any type `std::map` can order). When the device
/// has no memory left to instantiate a new graph, it frees the least recently used ones until the new one fits; they
/// are captured again when used. Not thread-safe. All replays use one stream; the caller waits for them before
/// clearing or destroying the cache.
template <class Key> class ShapeGraphCache {
public:
    ShapeGraphCache() = default;
    ~ShapeGraphCache() { clear(); }
    ShapeGraphCache(const ShapeGraphCache&) = delete;
    ShapeGraphCache& operator=(const ShapeGraphCache&) = delete;

    /// Gets the graph of `key` (null when none is captured) and marks it as just used.
    cudaGraphExec_t find(const Key& key) {
        const auto it = graphs_.find(key);
        if (it == graphs_.end())
            return nullptr;
        it->second.used = ++tick_;
        return it->second.exec;
    }

    /// Launches the graph of `key` on `stream`, first capturing the launches `body` makes on `stream` when the
    /// shape has none yet. `body` runs only when it is captured, so it must only launch work.
    template <class F> bool run(const Key& key, cudaStream_t stream, F&& body, std::string& err) {
        cudaGraphExec_t ex = find(key);
        if (!ex && !(ex = capture(key, stream, [&] { body(); return true; }, "graph capture: ", err)))
            return false;
        const cudaError_t le = cudaGraphLaunch(ex, stream);
        if (le != cudaSuccess)
            err = std::string("graph launch: ") + cudaGetErrorString(le);
        return le == cudaSuccess;
    }

    /// Records the launches `record` makes on `stream` (it returns false, with `err` set, when recording failed) and
    /// keeps them as the graph of `key` (see `instantiate`), which it returns; null on failure, with `err` starting
    /// with `what` unless `record` failed.
    template <class F>
    cudaGraphExec_t capture(const Key& key, cudaStream_t stream, F&& record, const char* what, std::string& err) {
        if (const cudaGraphExec_t ex = find(key))
            return ex;
        cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            err = std::string(what) + cudaGetErrorString(e);
            return nullptr;
        }
        cudaGraph_t graph = nullptr;
        bool recorded;
        try {
            recorded = record();
        } catch (...) {
            (void) cudaStreamEndCapture(stream, &graph);
            if (graph)
                (void) cudaGraphDestroy(graph);
            throw;
        }
        e = cudaStreamEndCapture(stream, &graph);
        if (recorded && e != cudaSuccess)
            err = std::string(what) + cudaGetErrorString(e);
        const cudaGraphExec_t ex = recorded && e == cudaSuccess ? instantiate(key, graph, stream, what, err) : nullptr;
        if (graph)
            (void) cudaGraphDestroy(graph);
        return ex;
    }

    /// Gets the graph of `key`, instantiating `graph` if absent. While the device has no
    /// memory for it, frees the least recently used unpinned graph (after waiting for `stream`, which may still
    /// replay it) and tries again. Null on failure, with `err` starting with `what`.
    cudaGraphExec_t instantiate(const Key& key, cudaGraph_t graph, cudaStream_t stream, const char* what,
                                std::string& err) {
        if (const cudaGraphExec_t ex = find(key))
            return ex;
        // (into a local: what a failed instantiate leaves in its output is not specified)
        cudaGraphExec_t got = nullptr;
        cudaError_t ie = cudaGraphInstantiate(&got, graph, 0);
        int freed = 0;
        while (ie == cudaErrorMemoryAllocation && evict_oldest(stream)) {
            (void) cudaGetLastError();
            ++freed;
            ie = cudaGraphInstantiate(&got, graph, 0);
        }
        if (freed > 0)
            std::fprintf(stderr, "strata: no VRAM for a new window graph: freed the graphs of %d other window shapes, "
                                 "%zu kept (they are captured again when used)\n", freed, graphs_.size());
        if (ie != cudaSuccess) {
            (void) cudaGetLastError();
            err = std::string(what) + cudaGetErrorString(ie);
            return nullptr;
        }
        graphs_[key] = {got, ++tick_};
        return got;
    }

    /// Calls `f`, which returns whether it succeeded, while the graphs it finds or captures cannot be freed: for a
    /// caller that captures several graphs and then replays them all. Nested calls keep the outer call's pins.
    template <class F> bool pinning(F&& f) {
        /// Restores the enclosing scope's eviction boundary, including when the callable throws.
        struct RestorePins {
            uint64_t& target;
            uint64_t saved;
            ~RestorePins() { target = saved; }
        } restore{pinned_from_, pinned_from_};
        if (pinned_from_ == UINT64_MAX)
            pinned_from_ = tick_ + 1;
        return f();
    }

    /// Frees the least recently used unpinned graph after waiting for `stream`, and says whether it did.
    bool evict_oldest(cudaStream_t stream) {
        auto old = graphs_.end();
        for (auto it = graphs_.begin(); it != graphs_.end(); ++it)
            if (it->second.used < pinned_from_ && (old == graphs_.end() || it->second.used < old->second.used))
                old = it;
        if (old == graphs_.end() || cudaStreamSynchronize(stream) != cudaSuccess)
            return false;
        (void) cudaGraphExecDestroy(old->second.exec);
        graphs_.erase(old);
        return true;
    }

    /// Destroys every graph. The caller makes sure no replay is still running.
    void clear() {
        for (auto& kv : graphs_)
            (void) cudaGraphExecDestroy(kv.second.exec);
        graphs_.clear();
    }

    /// Gets the number of shapes that have a graph.
    size_t size() const { return graphs_.size(); }

private:
    /// A captured graph and when it was last used (a value of `tick_`).
    struct Entry {
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    std::map<Key, Entry> graphs_;
    /// Counts uses of the cache's graphs; orders them from least to most recently used.
    uint64_t tick_ = 0;
    /// While `pinning` runs: the first tick of its graphs, which `evict_oldest` keeps.
    uint64_t pinned_from_ = UINT64_MAX;
};

}  // namespace strata::core
