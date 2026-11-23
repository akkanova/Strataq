# src/common/sample/

The sampler chain on the host side, logit bias (`logit_bias.hpp`, from `src/core/`), and the seed of each sampled
row (`row_runs.hpp`: position counters independent of earlier draws and batch layout). Headers:
`include/strata/common/sample/`.

The rest is filled when the engines' sampling code is merged: DeepSeek has its own sampler kernels
(`include/strata/models/deepseekv41/sampler.hpp`, merged in **B2**).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
