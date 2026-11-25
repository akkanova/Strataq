// include/strata/common/kernels/draft_rule.hpp - how a verify row of a sampled request judges the drafter's guess:
// the rule (a field of SamplerParams), its setting's text (--draft-rule, CHANGE_SETTINGS draft_rule=) and
// pick_verified_token, the one decision both paged engines' verifiers make, on the GPU or the host.
//
// At a verify row, p is the request's sampler chain over the kept list (descending), d the draft the row judges and
// x the plain draw, the token decoding without drafts writes there.  The row writes d when the rule keeps it, else
// another token, so the caller's "count the leading rows whose token is the draft" is the same for every rule.
//   exact               d only when d == x: the answers are those of decoding without drafts.
//   speculative         rejection sampling (Leviathan et al. 2023, Chen et al. 2023): keep d with probability
//                       min(1, p(d) / q(d)), else draw from max(0, p - q) renormalised; no drafter list means q is a
//                       point mass at d.  Every token is still distributed as p, so the answers' quality is p's.
//   typical             Medusa's typical acceptance (arXiv 2401.10774): keep d when d == x or
//                       p(d) > min(kTypicalEpsilon, kTypicalDelta * exp(-H(p))), H in nats; else x.
//   top-<N>             keep d when d == x, p(d) >= p(x), or d is in p's top-N% nucleus (the tokens ranked above it
//                       hold less than N% of p); else x.  lower-temperature is top-0: p(d) >= p(x).
// The lossy rules (typical, top-<N>, lower-temperature) allow extra drafts and change the output distribution.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_DRAFT_HD __host__ __device__
#else
#define STRATA_DRAFT_HD
#endif

namespace strata::kernels {

/// Which drafts a verify row of a sampled request keeps (the rules are in the header comment).  A greedy request
/// always compares its drafts with the argmax, whatever the rule.
struct DraftRule {
    enum Kind : uint8_t { Exact, Speculative, Typical, Nucleus };
    /// Speculative by default: what sample_tokens_spec and DeepSeek's sample_verify judge with unless told otherwise.
    Kind kind = Speculative;
    uint8_t top_percent = 0;   ///< Nucleus: the N of top-N, 0 to 99; 0 is lower-temperature
};

/// Medusa's typical acceptance: the hard probability floor ε and the entropy-scaled floor δ.
inline constexpr double kTypicalEpsilon = 0.09, kTypicalDelta = 0.3;

/// The three uniforms in [0, 1) a verify row may use, each from the caller's own Philox stream.
struct DraftUniforms {
    float plain;    ///< the plain draw's (decoding without drafts draws with it at this position)
    float accept;   ///< speculative sampling's accept test
    float resid;    ///< speculative sampling's draw from the residual after a reject
};

/// Gets the index into `p[0, n)` of the inverse-CDF draw with `u`: the sampler chain's own pick.
STRATA_DRAFT_HD inline int draw_index(const double* p, int n, float u) {
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += p[i];
        if ((double) u < cum)
            return i;
    }
    return n > 0 ? n - 1 : 0;
}

/// Draws with `u` from max(0, p - q) renormalised over the kept list `ids` / `p` (n entries); with `point`, q is a
/// point mass at `d`, so the residual is p without d, else q_of(i) is the drafter's probability of ids[i].  Gets -1
/// when the residual has no mass.
template <class QOf>
STRATA_DRAFT_HD int draw_residual(const int* ids, const double* p, int n, int d, bool point, QOf&& q_of, float u) {
    auto weight = [&](int i) {
        double w = point ? (ids[i] == d ? 0.0 : p[i]) : p[i] - q_of(i);
        if (w < 0.0)
            w = 0.0;
        return w;
    };
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += weight(i);
    if (!(sum > 0.0))
        return -1;
    double cum = 0.0;
    int last = -1;
    for (int i = 0; i < n; ++i) {
        const double w = weight(i);
        if (!(w > 0.0))
            continue;
        last = i;
        cum += w / sum;
        if ((double) u < cum)
            return ids[i];
    }
    return ids[last];
}

/// Gets the token a verify row of a sampled request writes: its draft `d` when `rule` keeps it, else another one
/// (the speculative residual can also select d). `ids` / `p` holds the kept list after the request's sampler chain,
/// descending, with n entries summing to 1.
/// `qd`: the drafter's probability of d, <= 0 for a point mass; q_of(i): its probability of ids[i], read only when a
/// speculative reject has qd > 0. A row without a draft (d < 0) gets the plain draw.
/// `plain_on_empty_residual` preserves samplers that draw with the residual uniform when p and q coincide.
template <class QOf>
STRATA_DRAFT_HD int pick_verified_token(DraftRule rule, const int* ids, const double* p, int n, int d, double qd,
                                        QOf&& q_of, DraftUniforms u, bool plain_on_empty_residual = false) {
    if (d < 0 || rule.kind == DraftRule::Exact)
        return ids[draw_index(p, n, u.plain)];
    int di = -1;
    for (int i = 0; i < n && di < 0; ++i)
        if (ids[i] == d)
            di = i;
    const double pd = di >= 0 ? p[di] : 0.0;
    if (rule.kind == DraftRule::Speculative) {
        const bool point = !(qd > 0.0);
        if (pd > 0.0 && (point ? (double) u.accept < pd : (double) u.accept * qd < pd))
            return d;
        const int r = draw_residual(ids, p, n, d, point, q_of, u.resid);
        return r >= 0 ? r : plain_on_empty_residual ? ids[draw_index(p, n, u.resid)] : d;
    }
    const int xi = draw_index(p, n, u.plain);
    if (di < 0)
        return ids[xi];
    if (rule.kind == DraftRule::Typical) {
        double entropy = 0.0;
        for (int i = 0; i < n; ++i)
            if (p[i] > 0.0)
                entropy -= p[i] * log(p[i]);
        return di == xi || pd > fmin(kTypicalEpsilon, kTypicalDelta * exp(-entropy)) ? d : ids[xi];
    }
    double above = 0.0;
    for (int i = 0; i < di; ++i)
        above += p[i];
    return pd >= p[xi] || above < rule.top_percent / 100.0 ? d : ids[xi];
}

/// The texts parse_draft_rule reads, in words for an error message.
inline constexpr const char* kDraftRuleChoices = "exact, speculative, typical, lower-temperature or top-1 to top-99";

/// Reads a draft rule's setting text: exact, speculative, typical, lower-temperature or top-<N> (N from 1 to 99).
/// False, leaving `rule` alone, for any other text.
inline bool parse_draft_rule(const std::string& text, DraftRule& rule) {
    if (text == "exact" || text == "speculative" || text == "typical" || text == "lower-temperature") {
        rule = {text == "exact" ? DraftRule::Exact : text == "speculative" ? DraftRule::Speculative
                : text == "typical" ? DraftRule::Typical : DraftRule::Nucleus, 0};
        return true;
    }
    const std::string digits = text.rfind("top-", 0) == 0 ? text.substr(4) : "";
    if (digits.empty() || digits.size() > 2 || digits.find_first_not_of("0123456789") != std::string::npos ||
        digits[0] == '0')
        return false;
    rule = {DraftRule::Nucleus, (uint8_t) std::stoi(digits)};
    return true;
}

/// Gets the setting text parse_draft_rule reads back as `rule`.
inline std::string get_draft_rule_name(DraftRule rule) {
    switch (rule.kind) {
    case DraftRule::Exact:
        return "exact";
    case DraftRule::Speculative:
        return "speculative";
    case DraftRule::Typical:
        return "typical";
    default:
        return rule.top_percent > 0 ? "top-" + std::to_string(rule.top_percent) : "lower-temperature";
    }
}

}  // namespace strata::kernels

#undef STRATA_DRAFT_HD
