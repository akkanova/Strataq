#pragma once
#include "strata/base/hash.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::models {
/// Reads strata-vision records (int32 'SVE1', n, nx, ny, width, then n x width floats, in prompt order) and lays the
/// pictures over the prompt's <|image_pad|> runs, as mtmd does: text counts up by one, a picture's cells take
/// (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny). `positions` holds
/// [cell][3] for every cell the caller can reach (at least the prompt's); cells past the prompt count on by one.
/// `image_keys`: the server's key for each picture, else a picture's hash covers its embeddings. `image_hashes` gets
/// {first cell, hash} per picture, which the conversation cache compares. Used by Qwen3.6 and Qwen3.8.
template <class Id, class ImageKey>
bool read_prompt_images(const std::string& path, const std::vector<Id>& ids, int64_t width,
                        const std::vector<uint64_t>& image_keys, std::vector<int32_t>& positions,
                        std::vector<float>& img_rows, std::vector<const float*>& row_ptr,
                        std::vector<ImageKey>& image_hashes, std::string& ve) {
    constexpr int64_t kImagePad = 248056;
    const int64_t n = (int64_t) ids.size(), cells = (int64_t) positions.size() / 3;
    if (n > cells || cells == 0) {
        ve = "the image prompt exceeds the rotary position table";
        return false;
    }
    row_ptr.assign(ids.size(), nullptr);
    image_hashes.clear();
    const auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        positions[(size_t) c * 3] = (int32_t) t;
        positions[(size_t) c * 3 + 1] = (int32_t) h;
        positions[(size_t) c * 3 + 2] = (int32_t) w;
    };
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    img_rows.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) ve = "cannot open " + path;
    while (f && ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0)
            break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) width || hdr[1] > n) {
            ve = "bad embeddings file (expected strata-vision records of width " +
                 std::to_string((long long) width) + ")";
            break;
        }
        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        img_rows.resize(off + cnt);
        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) {
            ve = "short embeddings file";
            break;
        }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    if (f)
        std::fclose(f);
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (ve.empty() && i < n) {
        if (ids[(size_t) i] != kImagePad) {
            put(i, p, p, p);
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) {
            ve = "the prompt has more images than the embeddings file";
            break;
        }
        const Img& im = imgs[k++];
        {
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            uint64_t h = core::fnv1a64(grid, sizeof grid);
            if (image_keys.size() == imgs.size())
                h = core::fnv1a64(&image_keys[k - 1], sizeof(uint64_t), h);
            else
                h = core::fnv1a64(img_rows.data() + im.off, (size_t) im.n * (size_t) width * sizeof(float), h);
            image_hashes.push_back({i, h});
        }
        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                     " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) width;
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
    for (int64_t c = n; ve.empty() && c < cells; ++c)
        put(c, p + (c - n), p + (c - n), p + (c - n));
    return ve.empty();
}
}  // namespace strata::models
