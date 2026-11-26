# AnyJev × Qwen3.5-4B spike summary

- model: `Qwen/Qwen3.5-4B` (revision `None`)
- goldens: `scripts/spike_samples.jsonl` (12 rows, synthetic=True)
- anyjev 0.3.0 / transformers 5.22.1 / torch 2.6.0+cu124
- device: cuda, dtype: bfloat16, batch_size: 8
- backend: Qwen3_5ForCausalLM (32 layers, hidden 2560), tokenizer Qwen3_5Tokenizer
- question: 请判断该笔交易应如何处理：
- options: 通过, 人工复核, 拒绝
- prior: batch, max_perms: None, canonical_order: True

## Metrics

| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.750 | 0.783 | 0.298 | 0.472 | 0.757 | 0.719 | 1.0 | 0.250 | 0.0215 |
| L0 | 0.500 | 0.550 | 0.329 | 0.511 | 0.753 | 0.709 | 3.0 | 0.000 | 0.0618 |
| L0-noprior | 0.583 | 0.650 | 0.226 | 0.504 | 0.764 | 0.714 | 3.0 | 0.000 | 0.0608 |
| L0-cf | 0.667 | 0.733 | 0.314 | 0.568 | 1.007 | 0.812 | 3.0 | 0.000 | 0.0776 |
| L0-adaptive | 0.500 | 0.550 | 0.328 | 0.508 | 0.751 | 0.709 | 3.0 | 0.000 | 0.0615 |
| L1 | 0.500 | 0.550 | 0.146 | 0.478 | 0.713 | 0.644 | 3.0 | — | 0.0000 |

## Per-class TPR / TNR

### raw

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 3 | 1.000 | 1.000 |
| 人工复核 | 5 | 4 | 0.600 | 0.857 |
| 拒绝 | 4 | 5 | 0.750 | 0.750 |

### L0

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 4 | 1.000 | 0.889 |
| 人工复核 | 5 | 5 | 0.400 | 0.571 |
| 拒绝 | 4 | 3 | 0.250 | 0.750 |

### L0-noprior

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 4 | 1.000 | 0.889 |
| 人工复核 | 5 | 2 | 0.200 | 0.857 |
| 拒绝 | 4 | 6 | 0.750 | 0.625 |

### L0-cf

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 3 | 1.000 | 1.000 |
| 人工复核 | 5 | 1 | 0.200 | 1.000 |
| 拒绝 | 4 | 8 | 1.000 | 0.500 |

### L0-adaptive

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 4 | 1.000 | 0.889 |
| 人工复核 | 5 | 5 | 0.400 | 0.571 |
| 拒绝 | 4 | 3 | 0.250 | 0.750 |

### L1

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 通过 | 3 | 4 | 1.000 | 0.889 |
| 人工复核 | 5 | 5 | 0.400 | 0.571 |
| 拒绝 | 4 | 3 | 0.250 | 0.750 |

## Notes

- Samples are synthetic smoke-test rows, not the Phase 0 goldens; metrics are not a quality claim.
- AnyJev 0.3.0 implements raw and L0 only; L1 here is post-hoc temperature scaling fit in-sample.
- L0 uses a batch prior estimated from the same rows; no labels are used in the prior, but the eval batch's class composition does enter the correction.
- L0-noprior isolates permutation marginalization from prior correction; L0-cf uses content-free probes and costs extra forward passes.
- Timing is the median of --timing-repeats runs (fresh Decider each run) after a one-row warmup; it excludes model load and is not a serving benchmark.
- A 12-row run cannot support a promotion decision; it validates the API and pipeline.
