// include/strata/base/platform/env.hpp - reading the engines' settings from environment variables, and setting and
// removing them on Linux and on Windows, which has no setenv or unsetenv.
//
// The readers take an empty value as not set (on Windows, setting "" removes a variable) and read on every call.
#pragma once

#include <cstdlib>

namespace strata::platform {

/// Gets the value of the environment variable `name`, or nullptr when it is not set or empty.
inline const char* env_text(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? v : nullptr;
}

/// Tells whether the switch `name` is on: set to a value that does not start with '0' ("1", "yes").
inline bool is_env_on(const char* name) {
    const char* v = env_text(name);
    return v != nullptr && v[0] != '0';
}

/// Tells whether the switch `name` is set to a value that starts with '0', turning off a feature that is on by default.
inline bool is_env_off(const char* name) {
    const char* v = env_text(name);
    return v != nullptr && v[0] == '0';
}

/// Gets the setting `name` as a whole number (std::atoi, so text that is not a number is 0), or `dflt` when not set.
inline int env_int(const char* name, int dflt) {
    const char* v = env_text(name);
    return v != nullptr ? std::atoi(v) : dflt;
}

/// Gets a wide whole-number setting with std::atoll, or `dflt` when it is not set.
inline long long env_i64(const char* name, long long dflt) {
    const char* v = env_text(name);
    return v != nullptr ? std::atoll(v) : dflt;
}

/// Gets the setting `name` as a decimal number (std::atof), or `dflt` when it is not set.
inline double env_double(const char* name, double dflt) {
    const char* v = env_text(name);
    return v != nullptr ? std::atof(v) : dflt;
}

/// Gets the switch `name` as -1 when it is not set, else 1 when its number (as env_int reads it) is not 0, else 0.
inline int env_tri(const char* name) {
    const char* v = env_text(name);
    return v != nullptr ? std::atoi(v) != 0 : -1;
}

/// Sets the environment variable `name` to `value`; with `overwrite` false, a value already set is kept. On Windows an
/// empty `value` removes the variable.
inline void set_env(const char* name, const char* value, bool overwrite = true) {
    if (!overwrite && std::getenv(name) != nullptr)
        return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

/// Removes the environment variable `name`.
inline void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

}  // namespace strata::platform
