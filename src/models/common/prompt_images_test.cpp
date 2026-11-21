#include "strata/models/common/prompt_images.hpp"
#include "strata/common/state/conversation_cache.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

/// Checks grid positions, image identity and malformed records without a model or GPU.
int main() {
    const auto path = std::filesystem::temp_directory_path() / "strata_prompt_images_test.sve";
    const std::vector<int64_t> ids = {10, 248056, 248056, 248056, 248056, 11};
    const int32_t header[] = {0x31455653, 4, 2, 2, 2};
    const float pixels[] = {1, 2, 3, 4, 5, 6, 7, 8};
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
        out.write(reinterpret_cast<const char*>(pixels), sizeof pixels);
    }
    std::vector<int32_t> positions(30);
    std::vector<float> rows;
    std::vector<const float*> pointers;
    std::vector<strata::core::ConversationImageKey> hashes;
    std::string err;
    const auto read = [&](const std::vector<uint64_t>& keys) {
        err.clear();
        return strata::models::read_prompt_images(path.string(), ids, 2, keys, positions, rows, pointers,
                                                  hashes, err);
    };
    bool ok = read({});
    ok = ok && hashes.size() == 1 && hashes[0].start == 1 && pointers[0] == nullptr && pointers[1][0] == 1 &&
         pointers[4][1] == 8 && pointers[5] == nullptr;
    ok = ok && positions[3] == 1 && positions[6] == 1 && positions[7] == 1 && positions[8] == 2 &&
         positions[12] == 1 && positions[13] == 2 && positions[14] == 2 && positions[15] == 3 && positions[27] == 7;
    const uint64_t first = hashes[0].hash;
    ok = ok && read({123}) && hashes[0].hash != first;
    const uint64_t keyed = hashes[0].hash;
    {
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        const float changed = 9;
        out.seekp(sizeof header);
        out.write(reinterpret_cast<const char*>(&changed), sizeof changed);
    }
    ok = ok && read({}) && hashes[0].hash != first && read({123}) && hashes[0].hash == keyed;
    ok = ok && strata::core::fnv1a64("a", 1) == 0x44bd8ad473cd9906ull;
    ok = ok && strata::core::fnv1a64(nullptr, 0) == 1469598103934665603ull;
    ok = ok && strata::core::fnv1a64("bc", 2, strata::core::fnv1a64("a", 1)) ==
                   strata::core::fnv1a64("abc", 3);
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
        out.write(reinterpret_cast<const char*>(pixels), sizeof pixels);
    }
    const std::vector<int32_t> ids32 = {10, 248056, 248056, 248056, 248056, 11,
                                      248056, 248056, 248056, 248056, 12};
    std::vector<std::pair<int64_t, uint64_t>> starts;
    positions.resize(ids32.size() * 3);
    err.clear();
    ok = ok && strata::models::read_prompt_images(path.string(), ids32, 2, {123, 123}, positions, rows,
                                                  pointers, starts, err);
    ok = ok && starts.size() == 2 && starts[0].first == 1 && starts[1].first == 6 &&
         starts[0].second == keyed && starts[1].second == keyed && rows.size() == 16 &&
         pointers[1] == rows.data() && pointers[6] == rows.data() + 8 && pointers[9] == rows.data() + 14 &&
         pointers[10] == nullptr && positions[18] == 4 && positions[30] == 6;
    for (const auto& [bad_ids, expected] : std::vector<std::pair<std::vector<int32_t>, std::string>>{
             {{10, 248056, 11, 12, 13}, "image 1 has 4 rows but fewer <|image_pad|> tokens"},
             {{10, 248056, 248056, 248056, 248056, 11}, "the embeddings file has more images than the prompt"},
             {{10, 248056, 248056, 248056, 248056, 11, 248056, 248056, 248056, 248056},
              "the prompt cannot end in an image"}}) {
        err.clear();
        ok = ok && !strata::models::read_prompt_images(path.string(), bad_ids, 2, {}, positions, rows,
                                                       pointers, starts, err) && err == expected;
    }
    {
        std::ofstream out(path, std::ios::binary);
        const int32_t oversized[] = {0x31455653, 12, 3, 4, 2};
        out.write(reinterpret_cast<const char*>(oversized), sizeof oversized);
    }
    err.clear();
    ok = ok && !strata::models::read_prompt_images(path.string(), ids32, 2, {}, positions, rows,
                                                   pointers, starts, err) &&
         err == "bad embeddings file (expected strata-vision records of width 2)";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
    }
    std::filesystem::resize_file(path, sizeof header + 1);
    ok = ok && !read({}) && err == "short embeddings file";
    std::filesystem::remove(path);
    std::cout << (ok ? "prompt_images_test: passed\n" : "prompt_images_test: FAILED\n");
    return ok ? 0 : 1;
}
