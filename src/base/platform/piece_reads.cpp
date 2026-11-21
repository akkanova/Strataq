// src/base/platform/piece_reads.cpp - see include/strata/base/platform/piece_reads.hpp.
#include "strata/base/platform/piece_reads.hpp"
#include "strata/base/platform/memory_manager.hpp"
#include "strata/base/platform/os.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <linux/aio_abi.h>   // Linux's asynchronous file I/O ABI; no userspace library.
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32) || defined(__linux__)
namespace {

constexpr uint64_t kMerge = 32ull << 20;   ///< the limit for combining pieces; a single piece may be larger
#if defined(_WIN32)
// NTFS runs the unbuffered reads of a file one at a time while the file is mapped or cached anywhere (the engines
// map their GGUF shards): measured on a PCIe 5 drive, 512 KiB reads at queue depth 48 make 10.4 GB/s on a file
// nobody maps and 3.4 GB/s on a mapped one, 8 MiB reads 8.6 GB/s.  So windows less than kGap apart (the same role of
// nearby experts) are merged into requests of up to kMerge bytes.
constexpr uint64_t kGap = 1ull << 20;
#else
// No NTFS here: windows are merged only where they touch (the 4 KiB rounding of adjacent slices, consecutive experts
// of a role), never across a gap that would be read for nothing.
constexpr uint64_t kGap = 0;
#endif

/// A piece widened to whole sectors: `size` bytes from `a0`, the piece's `n` bytes `skip` bytes in, read by request
/// `req` starting `in_req` bytes into it.
struct Window { fd_t file; uint64_t a0, size, skip, n; uint8_t* dst; size_t req; uint64_t in_req; };
/// One read request: `size` bytes of `file` from `a0`, landing at `pos` of the staging buffer.
struct Req { fd_t file; uint64_t a0, size, pos; };

/// This thread's aligned staging buffer, plan and completion state, kept for its next batch.
struct Scratch {
    uint8_t* buf = nullptr;
    size_t cap = 0;
    std::vector<Window> win;
    std::vector<Req> req;
    std::vector<int64_t> got;   ///< bytes each request read (< 0: it failed)
#if defined(_WIN32)
    std::vector<HANDLE> ev;
    std::vector<OVERLAPPED> ov;
#else
    aio_context_t ctx = 0;
    std::vector<iocb> cb;
    std::vector<iocb*> cbp;
    std::vector<io_event> ev;
#endif
    ~Scratch() {
        release();
#if defined(_WIN32)
        for (HANDLE e : ev)
            CloseHandle(e);
#else
        if (ctx != 0)
            (void) syscall(SYS_io_destroy, ctx);
#endif
    }
    /// Grows the staging buffer to at least `total` bytes (whole MiB); false if the size cannot be allocated.
    bool reserve(uint64_t total) {
        if (total <= cap)
            return true;
        if (total > std::numeric_limits<size_t>::max() - ((1u << 20) - 1))
            return false;
        release();
        cap = (size_t) ((total + (1u << 20) - 1) >> 20 << 20);
        buf = (uint8_t*) anon_alloc(cap, -1, 1);
        if (buf == nullptr) {
            cap = 0;
            return false;
        }
        core::get_memory_manager().add(buf, cap, "ram", {"other", "SSD reads", "Expert read staging"});
        return true;
    }
    /// Frees the staging buffer.
    void release() {
        core::get_memory_manager().remove(buf);
        anon_free(buf, cap);
        buf = nullptr;
        cap = 0;
    }
};

uint64_t plan_requests(Scratch& sc, const FilePiece* pieces, size_t n);
bool issue_requests(Scratch& sc);

}  // namespace

bool read_pieces(const FilePiece* pieces, size_t n) {
    thread_local Scratch sc;
    for (size_t k = 0; k < n; ++k) {
        const FilePiece& p = pieces[k];
        if (p.bytes == 0)
            continue;
        if (p.dst == nullptr || p.off > (uint64_t) INT64_MAX - (kDirectAlign - 1) ||
            p.bytes > (uint64_t) INT64_MAX - (kDirectAlign - 1) - p.off)
            return false;
    }
    const uint64_t total = plan_requests(sc, pieces, n);
    if (sc.req.empty())
        return true;
#if defined(_WIN32)
    for (const Req& r : sc.req)
        if (r.size > MAXDWORD)
            return false;
#endif
    if (!sc.reserve(total) || !issue_requests(sc))
        return false;
    for (const Window& x : sc.win) {
        // A sector-aligned request may extend past EOF while all of the piece's bytes are present.
        if (sc.got[x.req] < (int64_t) (x.in_req + x.skip + x.n))
            return false;
        std::memcpy(x.dst, sc.buf + sc.req[x.req].pos + x.in_req + x.skip, (size_t) x.n);
    }
    return true;
}

namespace {

/// Fills sc.win and sc.req for the pieces; returns the staging bytes the requests need.
uint64_t plan_requests(Scratch& sc, const FilePiece* pieces, size_t n) {
    sc.win.clear();
    for (size_t k = 0; k < n; ++k) {
        const FilePiece& p = pieces[k];
        if (p.bytes == 0)
            continue;
        const uint64_t a0 = p.off & ~(kDirectAlign - 1);
        sc.win.push_back({p.file, a0, direct_span(p.off, p.bytes), p.off - a0, p.bytes, p.dst, 0, 0});
    }
    std::sort(sc.win.begin(), sc.win.end(), [](const Window& a, const Window& b) {
        return a.file != b.file ? a.file < b.file : a.a0 < b.a0;
    });
    sc.req.clear();
    uint64_t total = 0;
    for (Window& x : sc.win) {
        Req* q = sc.req.empty() || sc.req.back().file != x.file ? nullptr : &sc.req.back();
        const uint64_t xend = x.a0 + x.size;
        if (q == nullptr || (x.a0 > q->a0 + q->size && x.a0 - (q->a0 + q->size) >= kGap) ||
            std::max(q->a0 + q->size, xend) - q->a0 > kMerge) {
            sc.req.push_back({x.file, x.a0, 0, total});
            q = &sc.req.back();
        }
        x.req = sc.req.size() - 1;
        x.in_req = x.a0 - q->a0;
        if (xend > q->a0 + q->size) {
            const uint64_t extra = xend - (q->a0 + q->size);
            if (extra > std::numeric_limits<uint64_t>::max() - total)
                return std::numeric_limits<uint64_t>::max();
            total += extra;
            q->size = xend - q->a0;
        }
    }
    return total;
}

/// Issues every request of the plan and waits for all of them, setting sc.got; false when one could not be issued
/// or failed.
bool issue_requests(Scratch& sc) {
    sc.got.assign(sc.req.size(), -1);
#if defined(_WIN32)
    while (sc.ev.size() < sc.req.size()) {
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (e == nullptr)
            return false;
        sc.ev.push_back(e);
    }
    sc.ov.assign(sc.req.size(), OVERLAPPED{});
    // every request in flight before the first wait: the drive sees the whole batch as one queue
    size_t issued = 0;
    bool ok = true;
    for (size_t q = 0; q < sc.req.size(); ++q) {
        const Req& r = sc.req[q];
        OVERLAPPED& o = sc.ov[q];
        o.Offset = (DWORD) r.a0;
        o.OffsetHigh = (DWORD) (r.a0 >> 32);
        o.hEvent = sc.ev[q];
        if (!ReadFile((HANDLE) r.file, sc.buf + r.pos, (DWORD) r.size, nullptr, &o) &&
            GetLastError() != ERROR_IO_PENDING) {
            ok = false;
            break;
        }
        ++issued;
    }
    for (size_t q = 0; q < issued; ++q) {
        DWORD got = 0;
        if (GetOverlappedResult((HANDLE) sc.req[q].file, &sc.ov[q], &got, TRUE))
            sc.got[q] = got;
        else
            ok = false;
    }
    return ok && issued == sc.req.size();
#else
    // Each thread has its own context, so concurrent batches never see each other's events.
    constexpr long kDepth = 256;   // requests in flight per thread; a larger batch is read in waves of this many
    if (sc.ctx == 0 && syscall(SYS_io_setup, kDepth, &sc.ctx) != 0) {
        sc.ctx = 0;
        return false;   // no AIO (a seccomp profile, aio-max-nr)
    }
    sc.cb.resize(std::min<size_t>(sc.req.size(), (size_t) kDepth));
    sc.cbp.resize(sc.cb.size());
    sc.ev.resize(sc.cb.size());
    bool ok = true;
    for (size_t w0 = 0; ok && w0 < sc.req.size(); w0 += (size_t) kDepth) {
        const size_t wn = std::min<size_t>((size_t) kDepth, sc.req.size() - w0);
        for (size_t j = 0; j < wn; ++j) {
            const Req& r = sc.req[w0 + j];
            iocb& c = sc.cb[j];
            std::memset(&c, 0, sizeof c);
            c.aio_lio_opcode = IOCB_CMD_PREAD;
            c.aio_fildes = (uint32_t) r.file;
            c.aio_buf = (uint64_t) (uintptr_t) (sc.buf + r.pos);
            c.aio_nbytes = r.size;
            c.aio_offset = (int64_t) r.a0;
            c.aio_data = (uint64_t) (w0 + j);
            sc.cbp[j] = &c;
        }
        // every request of the wave queued; a refused submit ends the batch once what was queued has completed
        // (the kernel still writes into the buffer until then)
        size_t queued = 0;
        while (queued < wn) {
            const long r = syscall(SYS_io_submit, sc.ctx, (long) (wn - queued), sc.cbp.data() + queued);
            if (r > 0) {
                queued += (size_t) r;
                continue;
            }
            if (r < 0 && errno == EINTR)
                continue;
            ok = false;
            break;
        }
        for (size_t done = 0; done < queued;) {
            const long r = syscall(SYS_io_getevents, sc.ctx, 1L, (long) (queued - done), sc.ev.data(), nullptr);
            if (r < 0) {
                if (errno == EINTR)
                    continue;
                // the context is in an unknown state: drop it (io_destroy waits for what is still in flight)
                (void) syscall(SYS_io_destroy, sc.ctx);
                sc.ctx = 0;
                return false;
            }
            for (long j = 0; j < r; ++j) {
                const io_event& e = sc.ev[(size_t) j];
                sc.got[(size_t) e.data] = e.res;
                if (e.res < 0)
                    ok = false;
            }
            done += (size_t) r;
        }
    }
    return ok;
#endif
}

}  // namespace

#else
bool read_pieces(const FilePiece*, size_t) { return false; }
#endif

}  // namespace strata::platform
