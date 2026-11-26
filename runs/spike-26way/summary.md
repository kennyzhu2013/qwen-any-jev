# AnyJev × Qwen3.5-4B spike summary

- model: `Qwen/Qwen3.5-4B` (revision `None`)
- goldens: `scripts/spike_26way.jsonl` (3 rows, synthetic=True)
- anyjev 0.3.0 / transformers 5.22.1 / torch 2.6.0+cu124
- device: cuda, dtype: bfloat16, batch_size: 8
- backend: Qwen3_5ForCausalLM (32 layers, hidden 2560), tokenizer Qwen3_5Tokenizer
- question: 请从 26 个类别中选择最匹配的风险类型：
- options: 正常交易, 账户盗用, 盗卡, 促销滥用, 拒付, 商户异常, 套现, 刷单, 跨境大额, 合成身份, 洗钱, 赌博, 虚拟币, 电信诈骗, 木马盗刷, 钓鱼登录, 设备伪造, 地址异常, 频次异常, 金额异常, 夜间交易, 新户风险, 代理 IP, 群控设备, 内部欺诈, 其他
- prior: batch, max_perms: 4, canonical_order: True

## Metrics

| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 1.000 | 1.000 | 0.091 | 0.030 | 0.103 | 0.909 | 1.0 | — | 0.0468 |
| L0-noprior | 1.000 | 1.000 | 0.184 | 0.097 | 0.233 | 0.816 | 4.0 | — | 0.1413 |

## Per-class TPR / TNR

### raw

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常交易 | 1 | 1 | 1.000 | 1.000 |
| 账户盗用 | 1 | 1 | 1.000 | 1.000 |
| 盗卡 | 0 | 0 | — | 1.000 |
| 促销滥用 | 0 | 0 | — | 1.000 |
| 拒付 | 1 | 1 | 1.000 | 1.000 |
| 商户异常 | 0 | 0 | — | 1.000 |
| 套现 | 0 | 0 | — | 1.000 |
| 刷单 | 0 | 0 | — | 1.000 |
| 跨境大额 | 0 | 0 | — | 1.000 |
| 合成身份 | 0 | 0 | — | 1.000 |
| 洗钱 | 0 | 0 | — | 1.000 |
| 赌博 | 0 | 0 | — | 1.000 |
| 虚拟币 | 0 | 0 | — | 1.000 |
| 电信诈骗 | 0 | 0 | — | 1.000 |
| 木马盗刷 | 0 | 0 | — | 1.000 |
| 钓鱼登录 | 0 | 0 | — | 1.000 |
| 设备伪造 | 0 | 0 | — | 1.000 |
| 地址异常 | 0 | 0 | — | 1.000 |
| 频次异常 | 0 | 0 | — | 1.000 |
| 金额异常 | 0 | 0 | — | 1.000 |
| 夜间交易 | 0 | 0 | — | 1.000 |
| 新户风险 | 0 | 0 | — | 1.000 |
| 代理 IP | 0 | 0 | — | 1.000 |
| 群控设备 | 0 | 0 | — | 1.000 |
| 内部欺诈 | 0 | 0 | — | 1.000 |
| 其他 | 0 | 0 | — | 1.000 |

### L0-noprior

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常交易 | 1 | 1 | 1.000 | 1.000 |
| 账户盗用 | 1 | 1 | 1.000 | 1.000 |
| 盗卡 | 0 | 0 | — | 1.000 |
| 促销滥用 | 0 | 0 | — | 1.000 |
| 拒付 | 1 | 1 | 1.000 | 1.000 |
| 商户异常 | 0 | 0 | — | 1.000 |
| 套现 | 0 | 0 | — | 1.000 |
| 刷单 | 0 | 0 | — | 1.000 |
| 跨境大额 | 0 | 0 | — | 1.000 |
| 合成身份 | 0 | 0 | — | 1.000 |
| 洗钱 | 0 | 0 | — | 1.000 |
| 赌博 | 0 | 0 | — | 1.000 |
| 虚拟币 | 0 | 0 | — | 1.000 |
| 电信诈骗 | 0 | 0 | — | 1.000 |
| 木马盗刷 | 0 | 0 | — | 1.000 |
| 钓鱼登录 | 0 | 0 | — | 1.000 |
| 设备伪造 | 0 | 0 | — | 1.000 |
| 地址异常 | 0 | 0 | — | 1.000 |
| 频次异常 | 0 | 0 | — | 1.000 |
| 金额异常 | 0 | 0 | — | 1.000 |
| 夜间交易 | 0 | 0 | — | 1.000 |
| 新户风险 | 0 | 0 | — | 1.000 |
| 代理 IP | 0 | 0 | — | 1.000 |
| 群控设备 | 0 | 0 | — | 1.000 |
| 内部欺诈 | 0 | 0 | — | 1.000 |
| 其他 | 0 | 0 | — | 1.000 |

## Notes

- Samples are synthetic smoke-test rows, not the Phase 0 goldens; metrics are not a quality claim.
- AnyJev 0.3.0 implements raw and L0 only; L1 here is post-hoc temperature scaling fit in-sample.
- L0 uses a batch prior estimated from the same rows; no labels are used in the prior, but the eval batch's class composition does enter the correction.
- L0-noprior isolates permutation marginalization from prior correction; L0-cf uses content-free probes and costs extra forward passes.
- Timing is the median of --timing-repeats runs (fresh Decider each run) after a one-row warmup; it excludes model load and is not a serving benchmark.
- A 12-row run cannot support a promotion decision; it validates the API and pipeline.
