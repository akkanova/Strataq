// take_chunk_in_pieces (chunk_pieces.hpp): which pieces a GPU pool's chunk is taken in, and when it is outside.
#include "strata/common/serve/chunk_pieces.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::serve;

namespace {
void check(bool value, const char* description) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// A chunk's memory that answers each take with the next landing of a script, recording the piece sizes asked for.
struct Memory {
    std::vector<ChunkLanding> script;
    std::vector<uint64_t> taken;   ///< the piece size of each take
    int given_back = 0;

    /// Runs the pool policy against the scripted physical allocations.
    ChunkLanding take(uint64_t chunk_bytes) {
        return take_chunk_in_pieces(
            chunk_bytes,
            [this](uint64_t piece_bytes) {
                taken.push_back(piece_bytes);
                return script[taken.size() - 1];
            },
            [this] { ++given_back; });
    }
};
}  // namespace

int main() {
    const uint64_t chunk = 256ull << 20;
    {
        Memory m{{ChunkLanding::inside}};
        check(m.take(chunk) == ChunkLanding::inside && m.taken == std::vector<uint64_t>{chunk} && m.given_back == 0,
              "a chunk that fits as one piece stays one piece");
    }
    {
        Memory m{{ChunkLanding::none}};
        check(m.take(chunk) == ChunkLanding::none && m.taken.size() == 1 && m.given_back == 0,
              "a chunk the memory has no room for is not tried in pieces");
    }
    {
        Memory m{{ChunkLanding::outside, ChunkLanding::inside}};
        check(m.take(chunk) == ChunkLanding::inside && m.taken == std::vector<uint64_t>{chunk, kPieceBytes} &&
                  m.given_back == 1,
              "a chunk outside as one piece is given back and fits in small pieces");
    }
    {
        Memory m{{ChunkLanding::outside, ChunkLanding::outside}};
        check(m.take(chunk) == ChunkLanding::outside && m.taken.size() == 2 && m.given_back == 1,
              "a chunk some of whose small pieces land outside is outside");
    }
    {
        Memory m{{ChunkLanding::outside, ChunkLanding::none}};
        check(m.take(chunk) == ChunkLanding::none && m.given_back == 1, "small pieces the memory has no room for");
    }
    {
        Memory m{{ChunkLanding::outside}};
        check(m.take(kPieceBytes) == ChunkLanding::outside && m.taken.size() == 1 && m.given_back == 0,
              "a chunk no larger than a small piece stays one piece");
    }
    {
        Memory m{{ChunkLanding::outside}};
        check(m.take(kPieceBytes / 2) == ChunkLanding::outside && m.taken.size() == 1 && m.given_back == 0,
              "a chunk smaller than the fallback piece is not retried with a larger piece");
    }
    {
        const uint64_t partial_chunk = kPieceBytes + (2ull << 20);
        Memory m{{ChunkLanding::outside, ChunkLanding::inside, ChunkLanding::inside}};
        check(m.take(partial_chunk) == ChunkLanding::inside && m.take(partial_chunk) == ChunkLanding::inside &&
                  m.taken == std::vector<uint64_t>{partial_chunk, kPieceBytes, partial_chunk} && m.given_back == 1,
              "a partial last piece is allowed and the next chunk starts with a whole-piece attempt");
    }
    std::printf("chunk_pieces_test: ok\n");
    return 0;
}
