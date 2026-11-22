// Checks the draft rules' setting text and pick_verified_token's decisions on hand-computed distributions (CPU).
#include "strata/common/kernels/draft_rule.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using strata::kernels::DraftRule;
using strata::kernels::DraftUniforms;

int failures = 0;

/// Notes a failed condition without stopping, so that one run lists every failure.
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// A kept list after the sampler chain: token ids and their probabilities, descending.
struct List {
    std::vector<int> ids;
    std::vector<double> p;
};

/// One decision: the rule's text, the draft (and its drafter list q over `list`'s ids; empty: a point mass), the
/// uniforms and the token the row must write.
struct Case {
    const char* rule;
    const List* list;
    int draft;
    std::vector<double> q;
    DraftUniforms u;
    int want;
};

/// Gets the row's token for `c` through the shared decision.
int pick(const Case& c) {
    DraftRule rule;
    check(strata::kernels::parse_draft_rule(c.rule, rule), std::string("parse ") + c.rule);
    const List& l = *c.list;
    double qd = -1.0;
    for (size_t i = 0; i < c.q.size(); ++i)
        if (l.ids[i] == c.draft)
            qd = c.q[i];
    return strata::kernels::pick_verified_token(rule, l.ids.data(), l.p.data(), (int) l.ids.size(), c.draft, qd,
                                                [&](int i) { return c.q[(size_t) i]; }, c.u);
}

}  // namespace

int main() {
    for (const char* name : {"exact", "speculative", "typical", "lower-temperature", "top-1", "top-75", "top-90",
                             "top-98", "top-99"}) {
        DraftRule rule;
        check(strata::kernels::parse_draft_rule(name, rule) && strata::kernels::get_draft_rule_name(rule) == name,
              std::string("round trip ") + name);
    }
    for (const char* bad : {"", "top-", "top-0", "top-00", "top-05", "top-100", "top-7a", "top--5", "Exact", "top",
                            "nucleus", "lower_temperature", " exact"}) {
        DraftRule rule{DraftRule::Typical, 0};
        check(!strata::kernels::parse_draft_rule(bad, rule) && rule.kind == DraftRule::Typical,
              std::string("refuse '") + bad + "'");
    }

    // p = {0.7, 0.2, 0.1}: the plain draw is 10 for u < 0.7, 20 below 0.9, else 30.  H = 0.80 nats, so typical's
    // floor is min(0.09, 0.3 * 0.449) = 0.09: every listed token passes.
    const List three{{10, 20, 30}, {0.7, 0.2, 0.1}};
    // H = 1.164 nats: floor min(0.09, 0.0936) = 0.09, which 0.08 misses
    const List low{{1, 2, 3, 4}, {0.5, 0.3, 0.12, 0.08}};
    // H = 1.886 nats: floor min(0.09, 0.0455) = 0.0455, which 0.08 passes although it is below ε
    const List high{{1, 2, 3, 4, 5, 6, 7, 8}, {0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05}};
    const DraftUniforms x10{0.1f, 0.5f, 0.5f}, x20{0.8f, 0.5f, 0.5f}, x30{0.95f, 0.5f, 0.5f};
    const std::vector<Case> cases = {
        // no draft: the plain draw under every rule
        {"exact", &three, -1, {}, x20, 20}, {"speculative", &three, -1, {}, x30, 30},
        {"typical", &three, -1, {}, x10, 10}, {"top-90", &three, -1, {}, x20, 20},
        // exact: the plain draw, which keeps the draft only when they agree
        {"exact", &three, 10, {}, x10, 10}, {"exact", &three, 20, {}, x10, 10}, {"exact", &three, 30, {}, x30, 30},
        // speculative, point mass: keep with probability p(d), else p without d
        {"speculative", &three, 10, {}, x30, 10},                    // 0.5 < 0.7
        {"speculative", &three, 20, {}, x20, 10},                    // rejected; {0.7, 0.1} / 0.8 at 0.5
        {"speculative", &three, 20, {}, {0.1f, 0.1f, 0.5f}, 20},     // 0.1 < 0.2
        {"speculative", &three, 20, {}, {0.1f, 0.5f, 0.9f}, 30},     // rejected; 0.9 >= 0.875
        {"speculative", &three, 30, {}, x10, 10},                    // rejected; {0.7, 0.2} / 0.9 at 0.5
        {"speculative", &three, 99, {}, {0.1f, 0.0f, 0.95f}, 30},    // outside p: never kept; p itself at 0.95
        // speculative with q = {0.4, 0.5, 0.1}: keep with min(1, p/q), else max(0, p - q) = {0.3, 0, 0}
        {"speculative", &three, 10, {0.4, 0.5, 0.1}, {0.1f, 0.99f, 0.5f}, 10},
        {"speculative", &three, 20, {0.4, 0.5, 0.1}, {0.1f, 0.3f, 0.5f}, 20},   // 0.3 * 0.5 < 0.2
        {"speculative", &three, 20, {0.4, 0.5, 0.1}, {0.8f, 0.5f, 0.9f}, 10},   // 0.5 * 0.5 >= 0.2
        // typical
        {"typical", &three, 30, {}, x10, 30}, {"typical", &three, 99, {}, x20, 20},
        {"typical", &low, 4, {}, {0.1f, 0.0f, 0.0f}, 1}, {"typical", &low, 3, {}, {0.1f, 0.0f, 0.0f}, 3},
        {"typical", &low, 4, {}, {0.99f, 0.0f, 0.0f}, 4},            // the plain draw is the draft
        {"typical", &high, 5, {}, {0.1f, 0.0f, 0.0f}, 5},
        // lower temperature: p(d) >= p(x)
        {"lower-temperature", &three, 10, {}, x30, 10}, {"lower-temperature", &three, 20, {}, x10, 10},
        {"lower-temperature", &three, 20, {}, x30, 20}, {"lower-temperature", &three, 30, {}, x20, 20},
        {"lower-temperature", &three, 99, {}, x30, 30},
        // top-N: the tokens above d hold less than N%
        {"top-75", &three, 20, {}, x10, 20}, {"top-75", &three, 30, {}, x10, 10}, {"top-75", &three, 30, {}, x20, 20},
        {"top-70", &three, 20, {}, x10, 10},                          // 0.7 above it: not less than 70%
        {"top-71", &three, 20, {}, x10, 20}, {"top-89", &three, 30, {}, x10, 10}, {"top-91", &three, 30, {}, x10, 30},
        {"top-1", &three, 10, {}, x30, 10}, {"top-99", &three, 99, {}, x20, 20},
    };
    for (const Case& c : cases) {
        const int got = pick(c);
        check(got == c.want, std::string(c.rule) + " draft " + std::to_string(c.draft) + ": wrote " +
                             std::to_string(got) + ", want " + std::to_string(c.want));
    }
    // A float uniform rounded to 1 can reject even when p == q.
    const DraftRule spec{DraftRule::Speculative};
    const auto q = [&](int i) { return three.p[(size_t) i]; };
    const DraftUniforms rounded{0.1f, 1.0f, 0.95f};
    check(strata::kernels::pick_verified_token(spec, three.ids.data(), three.p.data(), 3, 10, 0.7, q, rounded) == 10,
          "empty residual keeps the shared sampler fallback");
    check(strata::kernels::pick_verified_token(spec, three.ids.data(), three.p.data(), 3, 10, 0.7, q, rounded,
                                               true) == 30,
          "empty residual keeps the DeepSeek sampler fallback");
    if (failures == 0)
        std::puts("draft_rule_test: passed");
    return failures == 0 ? 0 : 1;
}
