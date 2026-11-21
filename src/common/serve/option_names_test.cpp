#include "strata/common/serve/option_names.hpp"

#include <cstdio>
#include <cstring>
#include <initializer_list>

int main() {
    using strata::serve::get_primary_option;
    using strata::serve::kOptionAliases;
    int failures = 0;
    const auto check = [&](bool ok, const char* what, const char* name) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s: %s\n", what, name);
            ++failures;
        }
    };
    for (const auto& alias : kOptionAliases) {
        check(std::strcmp(get_primary_option(alias.old_name), alias.primary) == 0, "an old name gives its primary",
              alias.old_name);
        check(get_primary_option(alias.primary) == alias.primary, "a primary is not renamed (no chains)",
              alias.primary);
    }
    for (const char* name : {"--gpu", "--gpus", "--prefill-chunk", "--parallel", "--reserve"})
        check(get_primary_option(name) == name, "a name outside the table is kept", name);
    return failures == 0 ? 0 : 1;
}
