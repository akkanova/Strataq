#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::core {
/// Folds `n` bytes into a 64-bit FNV-1a hash; pass a previous result as `h` to continue it over more bytes. The default
/// seed is Strata's historical offset basis, used by picture identities and expert-file checksums.
inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i)
        h = (h ^ bytes[i]) * 1099511628211ull;
    return h;
}
}  // namespace strata::core
