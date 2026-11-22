// src/common/artifact/tensor_directory_test.cpp - TensorDirectory parses a pack listing (tensors.txt) line by line
// (no pack, no GPU): file rows by index, tensor rows with their dims, and malformed or unknown rows skipped; and
// for_each_round's staged pieces.
#include "strata/common/artifact/tensor_directory.hpp"

#include <cstdio>
#include <utility>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok)
        ++g_fail;
}
}  // namespace

int main() {
    strata::artifact::TensorDirectory toc;
    toc.add_line("file 1 model-00002.safetensors");
    toc.add_line("file 0 model-00001.safetensors\n");
    toc.add_line("t blk.0.attn_q.weight q8_0 1 4096 18874368 4096 4096\r");
    toc.add_line("t output_norm.weight f32 0 128 16384 4096");
    toc.add_line("t truncated.weight bf16 0 12");
    toc.add_line("# a comment");
    toc.add_line("");
    check(toc.files.size() == 2 && toc.files[0] == "model-00001.safetensors" &&
          toc.files[1] == "model-00002.safetensors", "file rows land at their index, in any order");
    const auto q = toc.tensors.find("blk.0.attn_q.weight");
    check(q != toc.tensors.end() && q->second.dtype == "q8_0" && q->second.file == 1 && q->second.off == 4096 &&
          q->second.bytes == 18874368, "a tensor row gives dtype, file, offset and size");
    check(q != toc.tensors.end() && q->second.shape == std::vector<int64_t>{4096, 4096},
          "a tensor row's trailing numbers are its shape (a CR is not a dim)");
    const auto n = toc.tensors.find("output_norm.weight");
    check(n != toc.tensors.end() && n->second.shape == std::vector<int64_t>{4096}, "a 1-D tensor keeps one dim");
    check(toc.tensors.count("truncated.weight") == 0 && toc.tensors.size() == 2, "rows short of a size are skipped");
    toc.add_line("file nope ignored.bin");
    toc.add_line("file -1 ignored.bin");
    toc.add_line("tag ignored bf16 0 0 2 1");
    toc.add_line("t overflow bf16 0 18446744073709551616 2 1");
    toc.add_line("t partial bf16 0 4oops 2 1");
    toc.add_line("t badshape bf16 0 0 2 nope");
    check(toc.files.size() == 2 && toc.tensors.size() == 2, "malformed fields and unknown tags are ignored");
    toc.add_line("\tt huge bf16 1 18446744073709551615 2 1\r\n");
    check(toc.tensors.at("huge").off == UINT64_MAX, "64-bit offsets and whitespace survive parsing");
    std::string err;
    check(!toc.load("no/such/tensors.txt", err) && !err.empty(), "a missing listing is an error naming the path");

    using strata::artifact::for_each_round;
    const uint64_t R = strata::artifact::kStageRound;
    std::vector<std::pair<uint64_t, uint64_t>> got;
    uint64_t stop_at = ~0ull;
    const auto record = [&](uint64_t done, uint64_t n) {
        got.emplace_back(done, n);
        return done < stop_at;
    };
    check(for_each_round(0, record) && got.empty(), "an empty tensor takes no rounds");
    check(for_each_round(2 * R + 1, record) && got == decltype(got){{0, R}, {R, R}, {2 * R, 1}},
          "rounds cover a tensor in order, the last one short");
    got.clear();
    stop_at = R;
    check(!for_each_round(3 * R, record) && got.size() == 2, "a failed round stops the rest");
    std::printf(g_fail == 0 ? "tensor_directory_test: ok\n" : "tensor_directory_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
