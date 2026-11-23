// include/strata/common/serve/option_names.hpp - the engines' older command-line names for one setting, so that each
// engine's parser knows only the primary name and every engine still takes the old one.
#pragma once

#include <cstring>

namespace strata::serve {

/// A command-line name that one engine used for a setting another engine (or a later version) calls `primary`.
struct OptionAlias {
    const char* old_name;
    const char* primary;
};

/// Deprecated spellings for engines that support the corresponding primary setting.
inline constexpr OptionAlias kOptionAliases[] = {
    {"--reserve-mib", "--vram-reserve-mib"},   // deprecated (strata-dsv41, strata-ds41): use --vram-reserve-mib
    {"--eos", "--eos-ids"},                    // deprecated (strata-dsv41, strata-ds41): use --eos-ids
    {"--slots", "--batch"},                    // deprecated (strata, strata-q38, strata-q36): use --batch
    {"--n", "--max-new"},                      // deprecated (strata-dsv41, strata-q36): use --max-new
    {"--ctx", "--max-context"},                // deprecated (strata-q36): use --max-context
    {"--chunk", "--prefill-chunk"},            // deprecated (strata-q36): use --prefill-chunk
    {"--prefill", "--prefill-chunk"},          // Qwen engines' name; setup still writes it (--prefill auto)
    // deprecated (strata, strata-q38, strata-q36): use --model. Setup, the Terminal and the app still write --native
    // because older installed engines know only that name (strata_common/user_args.py MODEL_FLAGS).
    {"--native", "--model"},
};

/// Gets the name an engine's parser compares command-line option `name` with: its primary if `name` is deprecated.
inline const char* get_primary_option(const char* name) {
    for (const OptionAlias& alias : kOptionAliases)
        if (std::strcmp(name, alias.old_name) == 0)
            return alias.primary;
    return name;
}

}  // namespace strata::serve
