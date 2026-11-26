# AnyJev × Qwen3.5-4B spike summary

- model: `Qwen/Qwen3.5-4B` (revision `None`)
- goldens: `scripts/spike_samples.jsonl` (12 rows, synthetic=True)
- anyjev 0.3.0 / transformers 5.22.1 / torch 2.6.0+cu124
- device: cuda, dtype: bfloat16, batch_size: 8
- question: 请判断该笔交易应如何处理：
- options: 通过, 人工复核, 拒绝
- prior: batch, max_perms: None, canonical_order: True

## Metrics

| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.750 | 0.783 | 0.298 | 0.472 | 0.757 | 0.719 | 1.0 | — | 0.1648 |
| L0 | 0.500 | 0.550 | 0.329 | 0.511 | 0.753 | 0.709 | 3.0 | — | 0.0618 |
| L0-adaptive | 0.500 | 0.550 | 0.328 | 0.508 | 0.751 | 0.709 | 3.0 | — | 0.0616 |
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
- L0 uses a batch prior estimated from the same rows; no labels are used in the prior.
- A 12-row run cannot support a promotion decision; it validates the API and pipeline.
