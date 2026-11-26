# AnyJev × Qwen3.5-4B spike summary

- model: `Qwen/Qwen3.5-4B` (revision `None`)
- goldens: `/tmp/dev_smoke_q1.jsonl` (67 rows, synthetic=False)
- anyjev 0.3.0 / transformers 5.22.1 / torch 2.6.0+cu124
- device: cuda, dtype: bfloat16, batch_size: 8
- backend: Qwen3_5ForCausalLM (32 layers, hidden 2560), tokenizer Qwen3_5Tokenizer
- question: 请根据以下复核规范判断通话风险：
【正常】银行或持牌机构正常贷款营销；助贷服务费合理（一般不超过10个点）且无第三方介入；外呼人员主动添加用户微信且无其他违规；正常客服、售后、普通商品营销；无虚假身份、无包过承诺、无提前收费。
【涉诈】贷款中介要求第三方担保、到场、收款、背调或包装资料（ab贷）；引导用户添加第三方微信；以年报补录、税务申报名义收费；荐股或引导进入投资平台/群、下载非官方APP；冒充公检法、律所、调解中心催收，威胁冻结/上门，要求对公转账；网贷退息退费；其他以收费或引流为目的的套路。
【原则】同时符合多个类型时，按风险更高的类型判断；仅有正常贷款推销、利率和额度介绍，无上述违规的，属正常。
请判断该通话对应的业务是否需要按涉诈风险处置：
- options: 正常, 涉诈
- prior: batch, max_perms: None, canonical_order: True

## Metrics

| config | acc | macro TPR | ECE | Brier | NLL | mean conf | mean shifts | reversal flip | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.522 | 0.504 | 0.308 | 0.662 | 1.012 | 0.830 | 1.0 | — | 0.1057 |

## Per-class TPR / TNR

### raw

| class | support | predicted | TPR | TNR |
|---|---:|---:|---:|---:|
| 正常 | 32 | 6 | 0.094 | 0.914 |
| 涉诈 | 35 | 61 | 0.914 | 0.094 |

## Notes

- Samples are synthetic smoke-test rows, not the Phase 0 goldens; metrics are not a quality claim.
- AnyJev 0.3.0 implements raw and L0 only; L1 here is post-hoc temperature scaling fit in-sample.
- L0 uses a batch prior estimated from the same rows; no labels are used in the prior, but the eval batch's class composition does enter the correction.
- L0-noprior isolates permutation marginalization from prior correction; L0-cf uses content-free probes and costs extra forward passes.
- Timing is the median of --timing-repeats runs (fresh Decider each run) after a one-row warmup; it excludes model load and is not a serving benchmark.
- A 12-row run cannot support a promotion decision; it validates the API and pipeline.
