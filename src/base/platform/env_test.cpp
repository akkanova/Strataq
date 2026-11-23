// src/base/platform/env_test.cpp - the settings readers of strata/base/platform/env.hpp (CPU only).
#include "strata/base/platform/env.hpp"

#include <cstdio>
#include <cstring>

int main() {
    using namespace strata::platform;
    const char* k = "STRATA_ENV_TEST_VALUE";
    struct Case { const char* value; bool on, off; int tri, num; double dec; };   // value nullptr: not set
    const Case cases[] = {{nullptr, false, false, -1, 7, 2.5}, {"", false, false, -1, 7, 2.5},
                          {"1", true, false, 1, 1, 1.0},       {"0", false, true, 0, 0, 0.0},
                          {"00", false, true, 0, 0, 0.0},       {"01", false, true, 1, 1, 1.0},
                          {"-2", true, false, 1, -2, -2.0},    {" 1", true, false, 1, 1, 1.0},
                          {"12junk", true, false, 1, 12, 12.0},
                          {"12", true, false, 1, 12, 12.0},    {"yes", true, false, 0, 0, 0.0},
                          {"0.75", false, true, 0, 0, 0.75}};
    int failed = 0;
    for (const Case& c : cases) {
        if (c.value != nullptr)
            set_env(k, c.value);
        else
            unset_env(k);
        const bool ok = is_env_on(k) == c.on && is_env_off(k) == c.off && env_tri(k) == c.tri &&
                        env_int(k, 7) == c.num && env_i64(k, 7) == c.num && env_double(k, 2.5) == c.dec &&
                        (env_text(k) != nullptr) == (c.tri >= 0);
        if (!ok) {
            std::printf("FAIL: value \"%s\"\n", c.value != nullptr ? c.value : "(not set)");
            ++failed;
        }
    }
    set_env(k, "5", false);
    if (std::strcmp(env_text(k), "0.75") != 0) {
        std::printf("FAIL: set_env without overwrite replaced the value\n");
        ++failed;
    }
    set_env(k, "4294967296");
    if (env_i64(k, 7) != 4294967296LL) {
        std::printf("FAIL: wide integer was narrowed\n");
        ++failed;
    }
    unset_env(k);
    if (env_i64(k, 7) != 7)
        ++failed;
    std::printf("env_test: %d failed\n", failed);
    return failed == 0 ? 0 : 1;
}
