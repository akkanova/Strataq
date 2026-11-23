#pragma once
// A pack's table of contents (for each tensor name, the file, byte range, type and shape that hold it), and the
// rounds in which a loader moves a tensor's bytes through a host buffer.

#include <algorithm>
#include <cstdint>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::artifact {

/// Says where one tensor's bytes lie in a pack and how to interpret them.
struct TensorEntry {
    std::string dtype;            ///< Element or block type as the pack writer named it ("bf16", "q8_0", ...).
    int file = -1;               ///< Index into TensorDirectory::files.
    uint64_t off = 0, bytes = 0;  ///< Byte range within that file.
    std::vector<int64_t> shape;   ///< Dimensions, outermost first.
};

/// Maps tensor names to where their bytes lie, read from a listing of lines `file <i> <name>` and
/// `t <name> <dtype> <file> <off> <bytes> <dims...>` (DeepSeek's tensors.txt). Other lines are ignored.
struct TensorDirectory {
    std::vector<std::string> files;  ///< File names relative to the model folder, by index.
    std::unordered_map<std::string, TensorEntry> tensors;

    /// Reads the listing at `path`; returns false with `err` set when it cannot be opened or read.
    bool load(const std::string& path, std::string& err) {
        std::ifstream in(std::filesystem::u8path(path));
        if (!in) {
            err = path + ": not found";
            return false;
        }
        for (std::string line; std::getline(in, line);)
            add_line(line);
        if (in.bad()) {
            err = path + ": read failed";
            return false;
        }
        return true;
    }

    /// Adds the file or tensor that one listing line names; skips malformed lines.
    void add_line(std::string_view line) {
        const auto tag = take_token(line);
        const auto name = take_token(line);
        if (tag.empty() || name.empty())
            return;
        if (tag == "file") {
            int i = -1;
            auto index = name;
            const auto file = take_token(line);
            if (!read_integer(index, i) || i < 0 || file.empty())
                return;
            if ((int) files.size() <= i)
                files.resize((size_t) i + 1);
            files[(size_t) i] = file;
            return;
        }
        if (tag != "t")
            return;
        TensorEntry x;
        x.dtype = take_token(line);
        if (x.dtype.empty() || !read_integer(line, x.file) || !read_integer(line, x.off) ||
            !read_integer(line, x.bytes))
            return;
        while (!line.empty()) {
            if (line.find_first_not_of(" \t\r\n") == std::string_view::npos)
                break;
            int64_t dim = 0;
            if (!read_integer(line, dim))
                return;
            x.shape.push_back(dim);
        }
        tensors.emplace(name, std::move(x));
    }

private:
    /// Advances over one whitespace-delimited field without allocating temporary strings.
    static std::string_view take_token(std::string_view& line) {
        size_t first = 0;
        while (first < line.size() && line[first] <= ' ')
            ++first;
        line.remove_prefix(first);
        size_t end = 0;
        while (end < line.size() && line[end] > ' ')
            ++end;
        const auto token = line.substr(0, end);
        line.remove_prefix(end);
        return token;
    }

    /// Rejects missing, out-of-range or partially numeric fields in a listing row.
    template <class T> static bool read_integer(std::string_view& line, T& value) {
        const auto token = take_token(line);
        if (token.empty())
            return false;
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        return result.ec == std::errc{} && result.ptr == token.data() + token.size();
    }
};

/// Bytes of a tensor that a loader stages in host memory at a time on their way to a device.
constexpr uint64_t kStageRound = 8ull << 20;

/// Calls `step(done, n)` for consecutive pieces [done, done + n) of a `bytes`-long tensor, each at most kStageRound
/// long, and stops at the first that returns false; returns whether every piece succeeded.
template <class Step> bool for_each_round(uint64_t bytes, Step&& step) {
    for (uint64_t done = 0; done < bytes;) {
        const uint64_t n = std::min(bytes - done, kStageRound);
        if (!step(done, n))
            return false;
        done += n;
    }
    return true;
}

}  // namespace strata::artifact
