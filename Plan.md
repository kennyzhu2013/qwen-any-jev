# Qwen3.5-4B 风控决策蒸馏方案（基于 AnyJev）

> 版本：v3 · 2026-11-03
> v2 修订见文末；v3 依据 anyjev 0.3.0 源码与 M0 spike 修正 L1/L2/adaptive 的描述。

## 一、目标与核心思路

**目标**：以 AnyJev（Nokia Applied Research，Apache 2.0）为决策接口，把 Qwen3.5-4B 变成风控决策模型，并把 AnyJev 教师的「K 轮旋转去偏」压缩成学生的「单次前向」，在尽量保住决策质量与校准的前提下降低推理成本。

**先明确这不是什么**：

- 这不是 Tacit 式 self-distillation。Tacit 的教师是**开启 thinking 的同一模型**，学生学的是「把推理压缩成一次前向」；本方案的教师是 AnyJev 的 **K 轮选项旋转去偏（raw/L0）**，压缩的是「去偏计算」。两者教师不同、收益不同，不能混为一谈。
- 更准确的叫法是 **procedure distillation / inference-time-compute distillation**。如果后续要做 Tacit 式推理压缩，应单独立项（教师 thinking 开、学生 one-pass），不要和本方案混在一起。

**角色定义**：

| 角色 | 配置 | 输出 |
|------|------|------|
| 教师 | Qwen3.5-4B + AnyJev（raw/L0，K 轮旋转；可加自适应旋转停止） | `answer` + `probs` + `level` |
| 学生 | 同一底座 + LoRA，复用 AnyJev 读出头 | 同一 option 集上的一次前向分布 |

**核心原则：off-ramp first。** 在训练之前，先证明「不训练」的路径不够用。AnyJev 0.3.0 的 `Decider` 只实现 raw/L0，**不含 L1/L2**；所谓「线性头」需要我们在 `HFBackend.hidden_states()` 上自实现（一次 prefill + 一个闭式 head），它仍是 LoRA 的最强替代方案。自适应旋转（`adaptive_shifts`）只减少旋转轮数（2×–K× prefill），不是单次前向。只有这些路径都过不了门禁，LoRA 蒸馏才值得做。

---

## 二、AnyJev 机制与 Qwen3.5-4B 约束

### 2.1 AnyJev 关键事实（据公开文档，P0 spike 逐条核实）

| 事实 | 内容 | 对方案的影响 |
|------|------|--------------|
| 决策读取 | 单次 prefill，从决策位置的 next-token logits 读 typed decision；不生成、不解析 | 训练/推理都必须精确对齐决策位置 |
| Decision 字段 | `answer`、`probs`、`level`、`diagnostics` | `level` 的含义见下 |
| `level` | **源码事实**：`LEVELS = ("raw", "L0")`，`Decision.level` 注释只有 raw/L0；不是样本级置信度 | Phase 4 的 λ₂「level 辅助分类」删除 |
| L0 | K 个选项循环旋转边际化 + 标签先验校正；zero-label；`combine="logmean"`；`DEFAULT_LOG_MARGIN=8.5` | 解决位置/标签先验偏差，不等于部署概率校准 |
| L1 | **0.3.0 未实现**。公开文档提过「L0 + 温度缩放」，但代码里没有 | 有标签时由我们在 V2 外层做温度缩放（spike 的 `L1` 即此近似） |
| L2 | **0.3.0 未实现**，没有线性头。`HFBackend.hidden_states()` 可导出中间层 | 若要做，需自实现闭式线性 probe（100–300 标签）；仍是 LoRA 的替代方案 |
| adaptive | `Decider(adaptive_shifts=True)` 是**旋转轮数的自适应停止**（`adaptive_min_shifts=2`、`adaptive_target=0.01`），不是 CoT 升级；CoT 升级属于 `Tacit`，需要 AnyJev 自训 checkpoint | 成本从 K× 降到 ≥2×，仍非单次前向；不能把它写成「80% 单次 + 20% CoT」 |
| 选项上限 | letter readout 最多 26 个选项（`MAX_OPTIONS=26`） | 决策体系必须分层或拆分 |
| vLLM 读法 | Tacit 在 vLLM 上只从 top-20 logprobs 读标签 | 软标签/ECE 评估优先用 HF logits，别用截断后的 top-20 |
| 成熟度 | pre-alpha，PyPI 0.3.0，接口仍在更新 | 钉版本 + revision，外包一层 adapter |
| 一致率 | `Tacit` 的 adaptive disagreement 是自洽性认证（本方案不经过 Tacit） | 不能当准确率/校准的质量门禁 |

### 2.2 Qwen3.5-4B 约束

- **默认开启 thinking**。必须**直接传 `enable_thinking=False`**；传 `chat_template_kwargs={"enable_thinking": False}` 会被当成普通模板变量、静默忽略，仍然打开 `<think>`（M0 已实测）。关闭后模板尾部为 `<think>\n\n</think>\n\n`。
- 训练、推理、评估三处必须使用**同一个模板字符串/版本**，并固化 `template_hash`；Qwen3.5 的 Think/NoThink 训练模板带 `{% generation %}` 标记，优先使用。
- `target_modules` 的模块名以实际 `state_dict` 为准，不要直接沿用 Qwen3 的命名。
- 固定模型 revision，不用 `main`。

### 2.3 M0 spike 结果（2026-11-03，已执行）

环境：Python 3.12.0（源码编译，OpenSSL 1.1.1c）、torch 2.6.0+cu124、transformers 5.22.1、anyjev 0.3.0、A100-PCIE-40GB、bf16。产物：`runs/spike-m0-full/`、`runs/spike-26way/`；脚本 `scripts/spike_anyjev.py`，smoke 数据 `scripts/spike_samples.jsonl`、`scripts/spike_26way.jsonl`。

已验证：

- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` 实际加载 **`Qwen3_5ForCausalLM`**（config `Qwen3_5TextConfig`，32 层、hidden 2560），视觉权重被忽略、文本权重正常加载；tokenizer 为 `Qwen3_5Tokenizer`。
- A–Z 在 `Qwen3_5Tokenizer` 上均为单 token（26/26）；26 选项的 raw/L0 读法在 3 条合成样本上跑通。
- AnyJev 的 `_render` 直接传 `enable_thinking=False`，HFBackend 路径默认已关闭 thinking。

合成 12 条 3 分类 smoke 结果（**不是质量结论**）：

| config | acc | ECE | NLL | reversal flip | s/row | 相对 raw |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.750 | 0.298 | 0.757 | 0.250 | 0.0215 | 1.00× |
| L0（batch prior） | 0.500 | 0.329 | 0.753 | 0.000 | 0.0618 | 2.87× |
| L0-noprior | 0.583 | 0.226 | 0.764 | 0.000 | 0.0608 | 2.83× |
| L0-cf | 0.667 | 0.314 | 1.007 | 0.000 | 0.0776 | 3.61× |
| L0-adaptive | 0.500 | 0.328 | 0.751 | 0.000 | 0.0615 | 2.86× |
| L1（in-sample T=1.60） | 0.500 | 0.146 | 0.713 | — | — | — |

解读：

- **位置偏差**：raw 有 25% 的样本在选项逆序后改答案，L0 为 0；旋转确实消除了位置敏感，这是 M0 最有价值的正面结果。但 M0.5 真实二分类数据上 raw 只有 2.5%（3/120）翻转，说明 12 条合成样本高估了位置偏差；L0 的零翻转在真实任务上不是实质收益（详见 §2.4）。
- **prior 不能自评测批估计**：L0 的 batch prior 在 12 条上反而降准确率；用评测批自身分布估计 prior 会引入批组成偏差。部署时必须用独立校准集/部署先验，或先试 `content_free`/`none`。
- **adaptive 没有省**：K=3 时 12/12 都用满 3 个 shift（默认 stop margin=8.5 很严）；必须在 K=5–10 的真实决策集上重测。
- **成本**：L0 ≈ 2.9× raw prefill，content-free ≈ 3.6×；这是 off-ramp 的关键数字，需在真实 K 与真实 prompt 长度上重测。M0.5 实测 L0/CF/adaptive 均 ≈1.96× raw（content-free 按问题共享探测，不是逐行 3–4×）。
- **结论**：API/读法兼容性通过，可以进入 M0.5；但 L0 的质量收益与成本优势尚未证明，不能据这 12 条决定上线。

### 2.4 M0.5 真实数据评估（2026-11-03，已完成）

数据：`assets/0729复核人工确认数据.csv`，1447 条人工确认通话，单日（2026-07-29）。`label` 正常 160 / 涉诈 1287；`category` 仅涉诈行有值。备注中有 10 条明确写“低风险”、6 条写“高风险”，但全部仍标 `涉诈`，因此这里的 `涉诈` 按**需要按涉诈/违规处置**理解，不等同于刑事诈骗；风险等级没有系统标注，M0.5 不做风险等级门禁。

任务：主任务二分类 `正常` vs `涉诈`；辅任务为 top-6 类型条件分类。冻结产物：`eval/goldens.jsonl`、`eval/goldens_category.jsonl`、`eval/splits.json`、`eval/stats.json`。按 `(label, 粗类型)` 分层 40/20/40 分 cal/dev/gate；gate 不参与任何选择。

对比：`raw` / `l0-noprior` / `l0-batch` / `l0-cf` / `l0+固定 prior`（自实现，prior 从 cal 估计）/ 温度缩放（cal 拟合）/ 线性 probe（自实现，层、L2、类权重、阈值在 dev 选）。协议预估成本：raw 1× prefill，L0 约 2×，content-free 约 3–4×，probe 1× prefill + 线性头。**实测**：raw 0.1136 s/row，L0 各变体 0.2221–0.2230 s/row（1.96×），content-free 也是 1.96×（不是 3–4×，因为它按问题共享 content-free 探测，不是逐行）；probe 1× prefill + 线性头。

**预注册门禁（在查看 gate 之前固定）**：

1. probe balanced accuracy ≥ 0.70；
2. 涉诈 recall ≥ 0.95；
3. 正常 recall ≥ 0.50；
4. ECE ≤ 0.10；
5. 相对最佳非-probe baseline 的 balanced accuracy 提升 ≥ 5pp，且配对 bootstrap 95% CI 下界 > 0。

全部满足 → 不做 LoRA，以「基座 + 线性头」结束 off-ramp；否则 LoRA 仍需评估。单日数据、gate 仅 64 条正常样本，正常 recall 的 95% CI 约 ±12pp，结论只作为 M0.5 方向判断，不作为上线承诺。

**执行结果（2026-11-03）**

产物：`runs/m0.5/report.md`（完整报告）、`dev/results.json`、`selection.json`、`gate/results.json`、`frontier_dev.json`、`frontier_gate.json`、`reversal.json`。脚本：`eval/prepare.py`、`features.py`、`probe.py`、`harness.py`、`frontier.py`、`reversal.py`、`report.py`。

cal/dev 选择结果（gate 未参与）：

| config | cal bal | cal 涉诈 R | cal 正常 R | dev bal | dev 涉诈 R | dev 正常 R | dev ECE | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.536 | 0.963 | 0.109 | 0.541 | 0.957 | 0.125 | 0.087 | 0.1255 |
| L0 无 prior | 0.535 | 0.961 | 0.109 | 0.541 | 0.957 | 0.125 | 0.075 | 0.2452 |
| L0 batch prior | 0.595 | 0.831 | 0.359 | 0.575 | 0.837 | 0.312 | 0.061 | 0.2453 |
| L0 content-free | 0.530 | 0.967 | 0.094 | 0.545 | 0.965 | 0.125 | 0.059 | 0.2462 |
| L0 adaptive | 0.535 | 0.961 | 0.109 | 0.541 | 0.957 | 0.125 | 0.075 | 0.2454 |

- 线性 probe：dev 选 layer 24 / L2 1.0 / balanced → acc 0.830、bal **0.713**、涉诈 recall 0.864、正常 recall 0.562、ECE 0.041。
- 风险优先 probe：layer 8 / L2 0.1 / 无类权重 → 涉诈 recall **0.961**、正常 recall 0.281、bal 0.621。
- 固定 prior：raw 选 strength 0.5（dev bal 0.604，涉诈 0.926 / 正常 0.281）；L0 选 strength 1.0（dev bal 0.592，涉诈 0.778 / 正常 0.406）。温度：raw 1.65、L0 1.70（cal NLL）。
- dev 操作点前沿（fraud≥0.95 下最高正常 recall）：raw 0.156、L0 batch 0.156、probe 0.219、probe 风险优先 0.281 —— **没有配置达到 0.50**。

gate 结果（冻结后只评分一次，578 条：涉诈 514 / 正常 64）：

| config | acc | bal | 涉诈 R | 正常 R | 涉诈 P | PR-AUC | ECE | s/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.879 | 0.556 | 0.971 | 0.141 | 0.901 | 0.943 | 0.054 | 0.1136 |
| raw+温度 | 0.879 | 0.556 | 0.971 | 0.141 | 0.901 | 0.943 | 0.053 | — |
| raw+固定 prior | 0.820 | 0.577 | 0.889 | 0.266 | 0.907 | 0.943 | 0.048 | — |
| L0 无 prior | 0.884 | 0.566 | 0.975 | 0.156 | 0.903 | 0.943 | 0.061 | 0.2221 |
| L0+固定 prior | 0.732 | 0.603 | 0.768 | 0.438 | 0.917 | 0.943 | 0.083 | — |
| L0 batch prior | 0.773 | 0.613 | 0.819 | 0.406 | 0.917 | 0.943 | 0.059 | 0.2224 |
| L0 content-free | 0.888 | 0.561 | 0.981 | 0.141 | 0.902 | 0.943 | 0.055 | 0.2230 |
| **线性 probe** | 0.846 | **0.742** | **0.875** | **0.609** | 0.947 | **0.979** | 0.065 | 1× prefill |
| probe 风险优先 | 0.879 | 0.569 | 0.967 | 0.172 | 0.904 | 0.964 | 0.044 | 1× prefill |
| probe@FPR5% | 0.344 | 0.631 | 0.263 | 1.000 | 1.000 | 0.979 | — | 1× prefill |
| probe@FPR10% | 0.687 | 0.762 | 0.665 | 0.859 | 0.974 | 0.979 | — | 1× prefill |

**门禁判定：未通过。** 冻结 probe 的 bal 0.742（≥0.70 ✅）、正常 recall 0.609（≥0.50 ✅）、ECE 0.065（≤0.10 ✅）、相对最佳非-probe baseline（raw+固定 prior，gate bal 0.577）提升 +0.165，配对 bootstrap 95% CI [0.086, 0.241] 下界 >0 ✅；**唯一失败项是涉诈 recall 0.875 < 0.95**。风险优先 probe 通过涉诈 recall（0.967），但正常 recall 0.172、bal 0.569，失败 3 项。baseline 按协议在 dev 上冻结；gate 上 L0+固定 prior 的 bal 反而更高（0.603），但不改选——以它为参照，probe 的 Δbal 仍为 +0.139（95% CI [0.056, 0.222]）。

关键限定：gate 上事后扫阈值到涉诈 recall 0.951 时，正常 recall 恰为 0.500、bal 0.726，看似满足全部条件；但该阈值不是预注册决策规则，且 dev 在同一约束下只有 0.219。正常类仅 32/64 条，这个 0.50 是边界波动，不能据此改判门禁通过。

**判读**

1. **L0 在二分类上没有质量收益**：`l0-noprior` 的 argmax 决策与 raw 几乎一致（gate bal 0.566 vs 0.556，涉诈 0.975 vs 0.971，正常 0.156 vs 0.141），成本 1.96×。`l0-cf` 同样；`l0-batch` 只是把操作点推向正常类（涉诈 0.819 / 正常 0.406），PR-AUC 与 raw 相同（0.943），没有新增可分性；`adaptive` 在 cal/dev 上与 `l0-noprior` 完全相同（k=2 时至少要 2 个 shift，省不了）。选项逆序测试（dev 120 条）也显示 raw 只有 2.5% 翻转（3/120；正常 1/32、涉诈 2/88），raw+固定 prior 3.3%，L0/L0+prior 为 0——raw 本来就接近顺序稳定，L0 的零翻转在本任务上不是实质收益。**L0/旋转路径在二分类出局。**
2. **固定 prior 只能在同一条 ROC 上移动操作点**：raw+prior@0.5 把正常 recall 从 0.141 提到 0.266，但涉诈 recall 降到 0.889；L0+prior@1.0 到 0.438 / 0.768。PR-AUC 不变（0.943），没有改变模型排序能力。
3. **probe 确实带来新的排序信号**：PR-AUC 0.943 → 0.979，ROC-AUC 0.694 → 0.860，gate bal +0.165；但它的收益主要是正常类（0.609），代价是涉诈 recall 0.875。风险优先点又把正常类压回 0.172。**没有一个稳健操作点同时满足 0.95 / 0.50。**
4. **成本结论**：L0/CF/adaptive 均 ≈1.96× raw；probe = 1× prefill + 线性头；LoRA 推理也是 1× prefill。**LoRA 相对 probe 没有延迟优势**，唯一的潜在收益是质量。
5. **数据/标签是主要瓶颈**：dev 上正常类与涉诈类的贷款词覆盖率 65.6%（21/32）vs 66.5%（171/257）；大量“正常”是合规贷款营销，而部分“涉诈”是违规催收/荐股。单看转写文本几乎不可分，标签很可能取决于通话外的业务归属/资质。任何 transcript-only 模型（probe 或 LoRA）都受这个上限约束。
6. **accuracy 不可用**：gate 涉诈占 88.9%，raw 的 accuracy 0.879 低于「全判涉诈」常数基线 0.889；必须看 bal 与两类 recall。

**决定**：按预注册协议，M0.5 未过门禁，**暂不启动 LoRA 蒸馏**。但这不构成「LoRA 值得做」的结论——失败点是涉诈 recall，而前沿显示当前表示在 0.95 约束下无法稳定保住正常类。下一步优先级：

1. **补标签**：风险等级（合规/低/中/高）、违规类型、业务归属/资质（是否持牌、是否授权营销、是否外包）、通话结果（是否成交/投诉/退款）。这些是区分「合规贷款营销」与「涉诈贷款营销」的必要信息。
2. **重定义操作点（已完成三路分流分析）**：在 dev 上按「复核预算 / 零漏放 / 平衡」选三路阈值，gate 只做对照评估、不参与选点（`runs/m0.5/triage.json`、报告 §3.2）。结论：**不存在可用的自动正常区间**——dev 前沿上所有点都关闭自动正常；gate 上复核容量 20%/30%/40%/50% 时，自动拦截涉诈 90.5%/83.7%/69.6%/65.0%，但正常类被误自动拒绝 40.6%/31.2%/17.2%/14.1%。「零漏放优先」可做到漏放 0、自动拦截 67.5%、复核 38.2%，代价仍是 15.6% 正常被自动拒绝。同一 `hi` 下正常误拒 dev 6.2% → gate 14.1%，说明 32 条 dev 正常类不足以冻结阈值。因此当前只能做「高置信涉诈自动拦截 + 其余全部转人工」，不能做自动正常；采用前需业务给出**误拒一个正常客户 vs 人工复核一条**的成本比，并把正常类扩到 300–500 条后重定阈值。另注意 gate 涉诈占比 88.9%，无约束的代价最小化会退化为“全部自动拒绝”（正常误拒成本仅 0.111×C_FP/行），必须先设正常误拒上限等硬约束。
3. **补评估集**：多日、时间外推拆分；正常类至少 300–500 条，把正常 recall 的 CI 从 ±12pp 压到 ±5pp 以内。
4. **复评 off-ramp**：用新标签重跑同一套 cal/dev/gate 与 probe。如果 probe 在新标签上过门禁 → 结束，不训 LoRA；如果 probe 在新标签上仍卡在门禁下且差距明确来自表示能力 → 再启动 M1 LoRA 小规模试点，用同一 gate 与 probe 对比。
5. 若业务坚持现在就要模型侧动作：只做小规模 LoRA 试点（同一数据、同一 gate），把它当作「表示能力上限」的实验，不作为上线候选；预期收益受标签确定性约束。

### 2.5 新标准 v2 标签重标（2026-11-05，已完成）

依据 `assets/spec.md`（最新质检复核标注规范）与 `assets/报备通话内容_违规场景分类.xlsx`，把旧的粗标签 `涉诈` 拆成 `is_violation` / `is_fraud` 两层，并补 `risk_level` / `scene_category` / `scene_subtype`。

- **标签拆分**：旧 `涉诈` 1287 条 = 4.2 涉诈 864 条 + 仅 4.1 违规 423 条；旧 `正常` 160 条保持合规（其中 15 条命中新标准风险信号，写入 `proposed_*` 字段并单列审核，不静默改写人工结论）。
- **242 条专项复核（已完成）**：human_0729 135 + agent_1022 107 全部逐条复核并回填（`scripts/backfill_spec_review.py`），结论文件 `runs/labels_v2/review_queue_复核结论.csv`。132 条维持、110 条改判、终判涉诈 8 条（均高风险）；结论来源 rule_engine 101 / 人工接受引擎 85 / 人工裁定 56。复核报告见 `runs/labels_v2/spec_review/复核报告.md`。
- **10.22 人批复核合并（2026-11-05）**：`assets/10.22agent和人工复核结果.xlsx` 的 6353 条人工复核覆盖原 `cq_agent_*` LLM/启发式银标签。6246 条采用人批复核（`label_source=human_1022`）；107 条与专项复核重叠，保留 spec 裁定。人批复核将 4.2 涉诈从 2 条校正为 50 条（39 条 ab贷 + 7 条年报补录收费 + 4 条引导加第三方微信，均高风险）。两套口径在 107 条上仍有 19 条风险、17 条场景、2 条涉诈分歧，清单见 `runs/labels_v2/spec_review/conflicts_human1022_vs_spec.csv`；当前按「专项 spec 裁定优先」，可改为一律以人批复核为准。
- **重庆正常数据合并（2026-11-05）**：`assets/重庆行业卡人工复核正常数据.xlsx` 1582 条，精确重复 1 条（`2026080216590008801197`）去除后 1581 条作为 `human_cq_normal` cohort 并入，全部合规/正常，无 id 或近重复簇与既有数据冲突。补齐正常类后 gate 正常样本达 695 条（human_cq_normal 632 + human_0729 63）。
- **最终分布（9381 条）**：cohort human_0729 1447 / agent_1022 6353 / human_cq_normal 1581；`is_fraud` 919（9.8%）/ 非涉诈 8462；`is_violation` 7633 / 合规 1748；风险 高风险 1285 / 低风险 6291 / 中风险 57 / 合规 1748；`needs_review=0`。拆分 cal/dev/gate = 3755/1876/3750，按近重复簇整簇分配（8586 个簇、91 个多行簇、最大 643 条、跨 split 桥接 0）；human_0729 的 1447 条沿用冻结拆分。
- **合并脚本**：`eval/merge_human_review.py`（默认 dry-run，`--write` 落盘），统计写入 `runs/labels_v2/human_review_merge_summary.json`。
- **产物**：`assets/0729复核人工确认数据_v2.csv`、`assets/复核数据_v2.csv`、`runs/labels_v2/summary_combined.json`、`runs/labels_v2/review_queue_*`、`runs/labels_v2/extra_splits.json`；评估集 `eval_v2/` 二分类 9381 条 / 4.1 大类 6714 条 / 4.2 类型 919 条，二分类目标为 `is_fraud`。备份：`assets/复核数据_v2.pre_human1022.bak.csv`、`assets/复核数据_v2.pre_specreview.bak.csv`、`runs/labels_v2/extra_splits.pre_normal.bak.json`、`runs/labels_v2/summary_combined.pre_review.bak.json`。
- **诊断（仅诊断，非 v2 模型）**：把 v1 冻结 probe 的概率按 `is_fraud` 重评分（只覆盖 `human_0729` cohort），合并后 gate 涉诈 recall 0.902、非涉诈 recall 0.300、precision 0.661、AUROC 0.765、AP 0.845、recall@FPR10 0.003；FP 仍集中在 `贷款相关`、`违规催收` 等仅 4.1 违规负类。说明旧 probe 学的是「违规需处置」而非 4.2 涉诈，**v1 的 M0.5 指标不能直接搬到新标准，需在 `eval_v2/` 上重训 probe 后复评**。

### 2.6 v2 probe 复评（2026-11-05，已完成）

在 9,381 条 v2 数据上用 `eval/task_v2.py` 的新 rubric 重新提特征并重训闭式线性 probe，协议与 v1 相同：层/L2/类权重只在 cal 上拟合、dev 上选择，gate 冻结后只评一次。

- **特征与选择**：`runs/v2/features_binary.npz`（9381×4×2560，最后一 token，GPU3 约 937s）；选中 layer 24、L2 0.1、class weight balanced。dev：bal 0.952、涉诈 recall 0.935、非涉诈 recall 0.969、ECE 0.011。
- **主门禁 human_0729 gate（578 条），argmax**：bal 0.891、涉诈 recall 0.934、非涉诈 recall 0.848、precision 0.903、PR-AUC 0.976、ECE 0.024（tp/fn/fp/tn = 325/23/35/195）。四项绝对门禁中**仅涉诈 recall 差 1.6pp**（0.934 < 0.95），严格判未通过。
- **dev 冻结的备选操作点**：FPR5（阈值 0.3075）在 gate 上 recall 0.951 / TNR 0.761 / bal 0.856 / ECE 0.029，FPR10（阈值 0.1403）recall 0.983 / TNR 0.626 / bal 0.804 / ECE 0.057；两者四项绝对门禁全部通过。直接按 dev 上「recall ≥ 0.95 前提下最大化 TNR」冻结的阈值 0.390（dev-R95）在 gate 上 recall 0.945 / TNR 0.809 / bal 0.877，仍差 0.5pp 未过——要过 0.95 只能把阈值压到约 0.31 以下。argmax 与各 FPR/recall 操作点的事前选择取决于误报成本，不能用 gate 反向选点。
- **采样不确定性**：主门禁涉诈类 348 条，argmax 的 23 个 FN 对应 recall 0.934，Wilson 95% CI 0.903–0.956；若真实 recall=0.95，观测到 ≤325/348 的概率约 10.8%，即点估计未过门禁但无法在 n=348 下与 0.95 统计区分（数据有近重复簇，区间偏窄）。门禁仍按预注册的点估计判未通过；要把 0.95 的判定稳定下来需要更多独立正例。
- **阈值迁移（重要）**：dev 选点的 FPR 在 gate 上明显放大——argmax 3.1% → 主门禁负类 15.2% / `human_cq_normal` 14.9%；FPR5 5% → 23.9% / 23.1%；FPR10 10% → 37.4% / 34.2%（全量 gate 因含大量易判负的 agent 行，FPR 仅 4.4% / 7.5% / 12.6%）。原因是 dev 与 gate 的负类构成/批次不同：主门禁负类中 167/230 是仅 4.1 违规，`human_cq_normal` 是另一批纯合规数据。`@FPR5`/`@FPR10` 只是 dev 选点目标，不是线上误报保证；阈值上线前必须用目标流量重新校准。
- **分 cohort（gate，argmax）**：agent_1022（2540 条，19 涉诈）recall 0.684 / TNR 0.992 / bal 0.838；human_cq_normal（632 条全正常）TNR 0.851（94 条误报）；全量 gate bal 0.938 / recall 0.921 / TNR 0.956。
- **对比 v1 probe（同 578 条 gate 行，按 v2 标签重评分）**：v1 bal 0.601 / recall 0.902 / TNR 0.300；v2−v1 Δbal +0.290 [0.251, 0.327]，Δacc +0.237 [0.202, 0.270]，McNemar b=16/c=153，p=3.03e-29。差值同时包含标签重定义与 rubric 变化，作诊断参考。
- **错误分析（全量 gate argmax，178 错）**：误报集中在仅 4.1 违规负类（正常 109、贷款相关 32）；漏报集中在 4.2 难例（引导贷款用户添加第三方微信 11、ab贷 9、证券投资类/年报补录/退息退费各 3）。
- **外部正常集复评（2026-11-20）**：重庆行业卡 `20261030-1102_normal_match` 去重后 8,268 条（旧标 + agent 均判正常，与 v2 零重合），冻结 probe 的实测 FPR：argmax 1.4%、dev-R95 2.5%、FPR5 4.1%、FPR10 13.4%；把 469 条旧标正常、agent 判违规/涉诈的冲突行也全部按正常计入（最保守），FPR5 仍只有 4.5%。说明 gate 主门禁 15.2% / `human_cq_normal` 14.9% 的高误报主要来自负类构成（167/230 为仅 4.1 违规、cq_normal 为 8 月批次），不是普遍的时间漂移；这批新流量支持 `@FPR5` 在正常流量上接近 dev 5% 目标。agent 负类可靠（重合核对：agent 正常/违规行的 v2 4.2 率仅 0.8%/0.1%），但 agent 涉诈 precision 仅 0.245，不能当金标。已生成 207 条 agent 涉诈、335 条 4.1 冲突、406 条旧涉诈-agent 正常的复核表 `runs/external_cq/review_queue_external.xlsx`。时间外推批次 11-04..11-13 去重后 3,966 条的描述性结果：agent 违规 3,487 条在 FPR5 的 probe 阳性率仅 0.8%，agent 正常 406 条 3.9%；agent 涉诈 73 条中 ab贷 14 条 probe 12/14 ≥0.5，引导加微 40 条 2/40、手机租赁 9 条 0/9。负类成本约 1% 已可评估。标签策略：14 条 forward ab贷 已确认为正例；其余 193 条按场景分层抽样 88 条估噪（`review_sample.xlsx`），仅当场景抽样精度点估计 ≥0.90 且 Wilson 下界 ≥0.70 时自动采纳剩余行，否则保持待定；probe 不参与标签采纳（避免高估 recall）。详见 `runs/external_cq/report.md` 与 `labeling_policy.md`。
- **Off-ramp 判断**：probe 在 v2 上有明确可分性（相对 v1 重评分 Δbal +0.290），但门禁与成本尚未同时闭合：argmax recall 差 1.6pp；FPR5 虽过 recall，在主门禁负类 / `human_cq_normal` 上误报约 23%，而在新外部正常集上仅 4.5%——误报成本高度依赖目标流量，必须按实际流量定。这不构成「表示能力不足」的证据，**暂不启动 M1 LoRA**；下一步是把操作点与误报成本交业务确认，并用目标流量校准阈值。若业务要求 argmax recall ≥0.95，优先补 4.2 难例（引导加微、ab贷）而不是直接上 LoRA。
- **产物**：`runs/v2/features_binary.npz`、`runs/v2/probe/`、`runs/v2/report.md`、`runs/v2/gate_results.json`；脚本 `eval/task_v2.py`、`eval/features.py --task v2`、`eval/report_v2.py`。

---

## 三、是否值得训：Off-ramp 决策树

按成本从低到高评估，任一层过门禁就停止：

| 层级 | 方案 | 成本 | 通过条件 |
|------|------|------|----------|
| 0 | 基座 + AnyJev raw/L0，K 轮 | K× prefill | 仅作参照，通常不满足单次前向目标 |
| 1 | **自实现闭式线性头**（文档称 L2，0.3.0 未提供） | 1× prefill + head，100–300 标签 | 在 goldens 上过晋升门禁 → **不训 LoRA** |
| 2 | L0 + 自适应旋转停止 | 2×–K× prefill（不是单次） | 仅降低旋转成本；质量达标才可用 |
| 3 | LoRA 蒸馏 | 1× prefill + 训练/维护成本 | 前两层都不过，且 K× 成本是真正瓶颈 |

**break-even**：只有当 `(K−1) × 单次 prefill 成本 × 线上流量 > 训练 + 维护成本` 时，蒸馏才划算。K 通常 5–10；流量低时可能永远回不了本，应把这条写进立项材料。

---

## 四、阶段设计

### Phase 0 · 评估先行（eval-harness-first）——必须先做

产出：

```
eval/
├── goldens.jsonl                    # 300–500 条带标签风控样本，版本化
├── graders/                         # 每个失败桶一个 grader
├── drift-suite.yaml                 # 200–500 条领域邻近题，冻结
└── baselines/
    ├── base-qwen3.5-4b.json         # 门禁令牌
    ├── anyjev-l0.json
    ├── anyjev-l0-temp.json           # 自实现温度缩放（文档称 L1）
    ├── anyjev-linear-probe.json      # 自实现线性头（文档称 L2，可选）
    └── anyjev-l0-adaptive.json       # 自适应旋转停止
```

- **错误分析**：对 ≥100 条真实风控 trace 开放编码，收敛为 4–8 个失败桶。建议桶：风险类型混淆、边界样本误判、概率过度/欠自信、输出格式不合规、拒绝决策不当、多标签漏判。
- **样本量（v1 的主要修正）**：≥100 是「开放编码」的下限，不是分桶统计的下限。建议 goldens 总量 **300–500**，**关键桶每桶 ≥50–100 个正例**；TPR/TNR 报 Wilson 置信区间，晋升比较用配对 McNemar 或 bootstrap。6 个桶、100 条样本时每桶约 17 条，TPR 的 95% CI 约 ±20–30pp，无法支撑「关键桶不退化」的判定。
- **代价矩阵前移**：在 Phase 0 定义漏判/误判的代价矩阵与操作点（如固定误判率下的漏判率），不要留到 Phase 5。
- **baseline 不止基座**：门禁令牌是 `base-qwen3.5-4b.json`，但参考系必须同时有 raw/L0/温度缩放/线性头/自适应旋转和延迟/成本（0.3.0 只直接提供 raw/L0，其余需自实现）。
- **grader**：typed decision 以确定性检查为主（格式、枚举、阈值）；很少需要 LLM-judge。若全部确定性，按 `eval-harness-first` 明确标注 judge N/A；若启用 judge，必须跨模型家族、报 TPR/TNR、钉快照。
- **drift suite**：200–500 条，冻结；与 goldens、训练场景做 ID + 语义双重污染检查。
- **退出条件**：goldens 已提交、每桶有 grader、judge 已校准或明确 N/A、drift suite 冻结、base + AnyJev 各模式 baseline 已写入。

### Phase 1 · 方法选型（finetuning-method-selection）

- 先执行第三章的 off-ramp。**自实现线性头若过门禁，本项目以「线性头 + 基座上线」结束，不进入训练阶段。**
- 数据形态是 demonstrations（prompt → 决策 + 概率），路线为 **SFT（LoRA）**；先不上 DPO/GRPO。若后续能拿到 cost-weighted 的可验证信号，再考虑 GRPO+RLVR。
- 显存：4B 的 **bf16 LoRA 在 24GB 上足够**，默认先试 LoRA；QLoRA（12–16GB）仅在 OOM 时启用。v1 把 QLoRA 写成起点，顺序应反过来。
- 框架：带 KL 的自定义蒸馏损失用不了 Unsloth 的编译 trainer；`messages` 形状 + `assistant_only_loss=True` 也应走 plain TRL/PEFT（Unsloth 2026.7.x 无该路径）。
- 模型目录 `references/model-catalog.md` 最后验证是 2026-07，尚无 Qwen3.5-4B；按技能要求先刷新目录再锁定底座。

### Phase 2 · 数据构造（dataset-curation）

1. **决策体系（先于数据生成）**
   - 明确单标签/多标签：AnyJev 的 Choice 是互斥选项；多标签风险需拆成多个 yes/no 或 score 决策。
   - 明确层级：风险大类 → 细类 → 路由；**单次决策 ≤26 个选项**（letter readout 上限）。
   - 固定 state 序列化格式（JSON / 自然语言模板），并纳入 `template_hash`。

2. **教师标注**
   - `Decider(HFBackend(qwen3.5-4b)).decide(...)`，`enable_thinking=False`；固定 K 轮旋转或 `adaptive_shifts`（记录每条实际 shifts）。
   - 教师配置优先在 goldens 上对比 raw/L0/温度缩放/线性头后再定，不要默认 L0；0.3.0 的 `Decider` 只接受 raw/L0。
   - **教师质量门禁前置**：按桶算教师 vs 人工的 TPR/TNR；教师弱的桶不用教师生成，改人工标注。

3. **软标签 schema（v1 缺失）** 每条至少存：

   ```
   scenario_id, options(顺序), teacher_probs(限制在同一 option 集上),
   K, rotation_order, seed, disagreement, level,
   anyjev_version, model_revision, template_hash, backend, dtype
   ```

4. **数据行**：保持 `messages` 格式；assistant 输出结构化决策；**概率作为软标签单独存字段，不写进文本**。

5. **质量控制**
   - 精确 + 语义双重去重（写明阈值）。
   - **「≥25% 真实数据」的口径**：按技能定义，real = 不是这个 student 为本任务生成的；教师输出严格来说可能满足该定义，从而使底线失效。本方案明确：**真实锚点 = 人工裁决的生产标签**；通用回放数据可计入 collapse guard，但不能替代人工标签来纠正教师偏差。
   - goldens 按 ID 永久排除出训练集。
   - 数据量：1,000 行作为 M3 pilot 下限；若以校准质量为目标，计划扩到 3,000–10,000 行，按失败桶定向采样。

6. **Dataset card 六字段**：来源、行数分布、真实/合成比、去重方法、模板字符串、packing 配置；另加教师配置、AnyJev 版本、模型 revision。

### Phase 3 · 训练（lora-qlora-recipes）

```yaml
target_modules: all-linear（以 Qwen3.5-4B 实际 state_dict 为准）
r: 16~32
lora_alpha: 2r
lora_dropout: 0
learning_rate: 2e-4（QLoRA 起点；bf16 LoRA 可从 1e-4~2e-4 起）
bf16: true（先确认硬件支持）
effective_batch: < 32
gradient_checkpointing: true
max_seq_len: <待定>
micro_batch / grad_accum: <待定>
epochs: 1~3
optimizer: adamw_8bit
seed: 固定
```

- **第一轮关掉 packing**：1k 行、短序列，padding 浪费可控；自定义损失需要在每条子样本中精确定位决策位置，packing 下位置映射是最容易出静默 bug 的地方。若要开，使用 varlen + 显式 position ids，并逐条解码验证。
- 数据按 Qwen3.5 的 **NoThink 模板**渲染后仅对 assistant 轮计算 loss（`assistant_only_loss=True`）；训练/推理/评估三处引用同一 `template_hash`。
- 开启 packing 时，必须人工抽检 5–10 条序列，确认边界、mask 和决策位置。
- checkpoint 选择用 dev 的桶指标 + ECE/Brier，不用 train loss；early stopping 与选型标准写进配置。
- 训练侧用 PEFT/TRL；产出仅保存 adapter（MB 级）。

### Phase 4 · 蒸馏损失设计

```
L = CE(决策 token, 硬标签)
  + λ₁ · τ² · Σ_i p_T(i) · log( p_T(i) / p_S(i) )
  + λ₂ · 校准正则（可选）
```

- **支撑集对齐（v1 的关键缺口）**：教师 probs 是限制在 option letter tokens 上再归一化的；学生必须复用 AnyJev 的读出头（同一决策位置、同一 option token 集）。如果学生生成 JSON 文本，KL 就在另一组 token 上计算，数学上不成立。**Phase 2 的 assistant 输出格式必须与此一致。**
- **KL 方向**：上面的写法是 `KL(p_T || p_S)`，与标准 KD 的软目标 CE 一致。v1 写的 `KL(p_S || p_T)` 方向相反，会让学生向教师 argmax 收缩、丢失教师的不确定性，和校准目标冲突。若要试反方向，必须作为对照实验。
- **温度**：教师/学生同温度 τ，损失乘 `τ²`；τ 做小网格。
- **硬标签来源**：有人工标签用人工；没有才用教师 argmax，并单独记录教师-人工分歧率。CE 与 KL 打架时，以人工标签为准。
- **后处理带来的分布偏移**：L0 的 probs 经过旋转平均和先验除法，学生原始 logits 未必能表示；实现时 clamp + 重归一化，并准备 post-hoc temperature scaling 兜底。
- **λ₂ / level**：已核实 0.3.0 的 `level` 只取 raw/L0（`LEVEL_ORDER` 仅两级），是校准层级而非置信度分档；**删除 λ₂ 的 level 分类**。若要置信度分档，先定义档位标签和损失。ECE 不可微，不要直接写进损失；用 Brier、熵正则或单独 head。
- **选型协议**：λ/τ 在 **dev** 上选，晋升门禁用另一套 goldens；网格做小（如 3×3×3），先在子集跑。用 held-out ECE/Brier 选型，不用 train loss。
- **先看教师熵**：教师 probs 接近 one-hot 时 KL 几乎不起作用；接近均匀时 KL 会主导。先画置信度分布再定量级。

### Phase 5 · 评估与迭代（eval-harness-first）

- 用**同一套 harness**跑每个 checkpoint，不另做宽松版。
- 指标：各桶 TPR/TNR（带 CI）、决策准确率、ECE/Brier、与教师全周期一致率、单次 vs K 轮延迟/成本、OOD/paraphrase 切片。
- **晋升门槛必须数值化**：例如「关键桶 TPR 的 95% CI 下界不低于 baseline − 5pp」，配合配对 McNemar/bootstrap；「不劣于 baseline」不能是无边界的点估计比较。
- **一致率不是质量指标**：学生可以和错误的教师高度一致，必须与人工 goldens 并列汇报。
- 延迟固定硬件、batch、prompt 长度，区分 prefill/decode，单独计 AnyJev 旋转开销。
- 上线后漂移监测：新失败模式回流错误分析；定义触发阈值和复查周期。

---

## 五、工程结构

```
risk-jev/
├── anyjev/                      # 固定 commit 的 AnyJev（或作为依赖钉版本）
├── eval/
│   ├── goldens.jsonl
│   ├── graders/
│   ├── drift-suite.yaml
│   └── baselines/
│       ├── base-qwen3.5-4b.json
│       ├── anyjev-l0.json
│       ├── anyjev-l0-temp.json
│       ├── anyjev-linear-probe.json
│       └── anyjev-l0-adaptive.json
├── data/
│   ├── scenarios.jsonl          # 风控场景与 route/标签定义
│   ├── teacher_outputs.jsonl    # 软标签 + 完整元数据
│   ├── train.jsonl
│   └── dataset_card.md
├── train/
│   ├── configs/lora_r16.yaml
│   └── train_self_distill.py
├── scripts/
│   └── spike_anyjev.py          # P0 可行性验证
├── runs/<run-id>/results.json
├── tests/
├── pyproject.toml / lockfile
└── Plan.md
```

- AnyJev 版本/commit 钉死；外面包一层 adapter，避免 `Decider` 接口变化扩散到训练与评估代码。
- 教师数据生成与评估使用同一个 backend、dtype，避免数值漂移。

---

## 六、主要风险

| 风险 | 应对 |
|------|------|
| 学生把教师的错误也蒸馏进去 | 训练前按桶校验教师质量；教师弱的桶改人工标注；记录教师-人工分歧率 |
| 单次前向丢失旋转的方差削减（鲁棒性不可蒸馏） | drift suite 加 paraphrase/OOD 切片，专门比较单次 vs K 轮的退化 |
| 概率软标签失真 / 先验错配 | L0 去的是标签先验，不是部署先验；对比 raw/L0/温度缩放/线性头的 ECE，必要时做部署先验校正 |
| 训练/推理模板不一致 | `template_hash` 集中配置，三处引用同一常量；显式 `enable_thinking=False` |
| 风控成本不对称 | 代价矩阵 + 操作点驱动阈值，不追求单一准确率 |
| 合成数据坍缩 | 守住 ≥25% 真实锚点（人工标签）+ 通用回放；dataset card 记录比例 |
| AnyJev pre-alpha 接口漂移 | 钉 commit/版本 + adapter 层 + 冒烟测试 |
| 后端/dtype 数值差异 | 教师数据生成与评估同 backend/dtype；软标签用 HF logits，不用 vLLM top-20 截断 |
| 26 选项上限 / letter 分词 | 决策分层，单次 ≤26；P0 确认 letter token 是单 token |
| 蒸馏不划算 | 先算 break-even；自实现线性头/自适应旋转通过则不做 LoRA |

---

## 七、里程碑与门禁

| 里程碑 | 内容 | 门禁 |
|--------|------|------|
| **M0** | P0 spike：AnyJev × Qwen3.5-4B（raw/L0/自适应旋转）、`level` 语义、26 选项、letter token、延迟 | **已通过**：决策可稳定读出、26/26 letter 单 token、模板可控；详见 2.3 |
| **M0.5** | Off-ramp 基线：自实现线性头 / 自适应旋转对比；prior 用独立校准集 | 线性头或自适应旋转过门禁 → 不做 LoRA；否则进入 M1 |
| **M1** | `eval/` + 全部 baseline（Phase 0 退出） | goldens 300–500、每桶 grader、drift 冻结、baseline 落盘 |
| **M2** | 教师软标签数据集 + dataset card | 教师分桶质量达标；真实锚点比例达标 |
| **M3** | LoRA + 自蒸馏损失，首个 run | 配置可复现；packing 检查通过；dev 指标可解释 |
| **M4** | 评估报告 + 晋升结论 | 非劣效边界 + 校准指标 + 成本收益 |

---

## 八、待确认事项

1. ~~`level` 字段的真实语义~~——P0 已核实：0.3.0 只有 raw/L0，是校准层级；L1/L2 未实现。
2. 风控决策是单标签还是多标签；是否需要层级与多决策拆分。
3. 教师用 raw/L0/温度缩放/线性头哪一档；部署先验是否需要重新校正。
4. `K` 用固定值还是自适应旋转停止（建议后者，记录每条实际 shifts）。
5. 线上流量与延迟预算——决定 break-even 和是否值得训。
6. 是否有足够的人工裁决标签作为「真实锚点」（goldens 之外另需一批）。
7. 训练/评估硬件（bf16 支持、单卡 24GB 是否足够、是否需要教师卡）。
8. L0 的 prior 用哪一批数据估计：独立校准集、部署先验，还是 `content_free`；不能再用评测批自身（M0 已暴露该偏差）。

---

## 附：v2 修改记录

1. **定位**：标题与正文从「自蒸馏」改为「决策蒸馏 / 过程蒸馏」，明确与 Tacit 式推理自蒸馏的区别。
2. **新增第三章 off-ramp**：L2 / adaptive 优先评估，L2 过门禁则不做 LoRA；补 break-even。
3. **Phase 0**：goldens 从 ≥100 提到 300–500，关键桶 ≥50–100 正例，补 CI/配对检验；baseline 增加 L0/L1/L2/adaptive；代价矩阵前移。
4. **新增 Qwen3.5-4B 约束**：强制 `enable_thinking=False`、模板 hash、固定 revision、target modules 核实。
5. **Phase 1**：bf16 LoRA 优先于 QLoRA；明确 plain TRL/PEFT（Unsloth 编译 trainer 不适用自定义 KL）。
6. **Phase 2**：新增决策体系约束（≤26 选项、多标签拆分）、软标签 schema、「真实数据」口径、教师质量前置门禁。
7. **Phase 3**：补全配置项；第一轮关 packing；NoThink 模板 + assistant-only loss；checkpoint 选择标准。
8. **Phase 4**：KL 方向改为 `KL(p_T || p_S)` 并补 `τ²`；强制支撑集对齐；删除/重定义 λ₂ 的 level 分类；dev 与门禁分离；教师熵检查。
9. **Phase 5**：晋升门槛数值化为非劣效边界；一致率与质量指标分离；延迟测量协议。
10. **风险与里程碑**：新增鲁棒性、先验错配、AnyJev 漂移、后端差异、break-even；新增 M0/M0.5 门禁与待确认事项。

## 附：v3 修改记录（依据 anyjev 0.3.0 源码）

1. **`level` 语义落地**：源码 `LEVELS = ("raw", "L0")`、`LEVEL_ORDER = {"raw": 0, "L0": 1}`，`Decision.level` 只有 raw/L0；Plan 中的 L1/L2 不再是 AnyJev 能力。
2. **L1 降级为自实现**：0.3.0 无温度缩放接口；spike 的 `L1` 是在 L0 probs 上后验拟合温度，且当前为 in-sample，不能当晋升证据。
3. **L2 降级为自实现**：0.3.0 无线形头；可行路径是 `HFBackend.hidden_states()` + 闭式 probe，成本仍是 1× prefill + head，保留为最强 off-ramp。
4. **adaptive 语义纠正**：`Decider(adaptive_shifts=True)` 是旋转轮数自适应停止（`adaptive_min_shifts=2`、`adaptive_target=0.01`），成本 2×–K×，**不是**「80% 单次 + 20% CoT」；CoT 升级属于 `Tacit`，需要 AnyJev 自训 checkpoint，不适用于本方案的 Qwen3.5-4B 教师。
5. **baseline/目录同步**：`anyjev-l1.json` → `anyjev-l0-temp.json`，`anyjev-l2.json` → `anyjev-linear-probe.json`，`anyjev-adaptive.json` → `anyjev-l0-adaptive.json`。
6. **M0/M0.5 与待确认事项同步**：删除「AnyJev 自带 L1/L2」的前提；`level` 问题标记为已核实。
7. **补 M0 实测结果（2.3 节）**：`Qwen3_5ForCausalLM` 加载成功；A–Z 26/26 单 token；`enable_thinking` 必须直接传参；raw 选项逆序翻转率 25%、L0 为 0；L0 成本 ≈2.9× raw；adaptive 在 K=3 无节省；batch prior 不能自评测批估计。
