# AnyJev × Qwen3.5-4B spike summary

- model: `Qwen/Qwen3.5-4B` (revision `None`)
- goldens: `/tmp/real_smoke.jsonl` (40 rows, synthetic=False)
- anyjev 0.3.0 / transformers 5.22.1 / torch 2.6.0+cu124
- device: cuda, dtype: bfloat16, batch_size: 8
- backend: Qwen3_5ForCausalLM (32 layers, hidden 2560), tokenizer Qwen3_5Tokenizer
- question: 请判断该通话对应的业务是否需要按涉诈风险处置：
- options: 正常, 涉诈
- prior: batch, max_perms: None, canonical_order: True

## Metrics

| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.650 | 0.573 | 0.400 | 0.708 | 1.533 | 0.874 | 1.0 | — | 0.1148 |
| L0 | 0.500 | 0.453 | 0.408 | 0.797 | 1.541 | 0.847 | 2.0 | — | 0.1858 |
| L0-noprior | 0.550 | 0.493 | 0.413 | 0.742 | 1.601 | 0.862 | 2.0 | — | 0.1864 |

## Per-class TPR / TNR

### raw

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常 | 15 | 7 | 0.267 | 0.880 |
| 涉诈 | 25 | 33 | 0.880 | 0.267 |

### L0

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常 | 15 | 13 | 0.267 | 0.640 |
| 涉诈 | 25 | 27 | 0.640 | 0.267 |

### L0-noprior

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常 | 15 | 11 | 0.267 | 0.720 |
| 涉诈 | 25 | 29 | 0.720 | 0.267 |

## Notes

- Samples are synthetic smoke-test rows, not the Phase 0 goldens; metrics are not a quality claim.
- AnyJev 0.3.0 implements raw and L0 only; L1 here is post-hoc temperature scaling fit in-sample.
- L0 uses a batch prior estimated from the same rows; no labels are used in the prior, but the eval batch's class composition does enter the correction.
- L0-noprior isolates permutation marginalization from prior correction; L0-cf uses content-free probes and costs extra forward passes.
- Timing is the median of --timing-repeats runs (fresh Decider each run) after a one-row warmup; it excludes model load and is not a serving benchmark.
- A 12-row run cannot support a promotion decision; it validates the API and pipeline.
