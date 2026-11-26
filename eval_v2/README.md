# 新标准评估集（v2，含 10.22 人工复核与重庆正常数据）

- 源文件：`assets/复核数据_v2.csv`（sha256 `89f6f902483e90a2…`）
- 生成链：`eval/merge_review_batch.py` → `scripts/backfill_spec_review.py` → `eval/merge_human_review.py` → `eval/prepare_v2.py`
- 样本：9381 条 = human_0729 1447 条 + agent_1022 6353 条 + human_cq_normal 1581 条。
- 拆分：cal 3755（human_0729 580 + agent_1022 2542 + human_cq_normal 633；涉诈 366） / dev 1876（human_0729 289 + agent_1022 1271 + human_cq_normal 316；涉诈 186） / gate 3750（human_0729 578 + agent_1022 2540 + human_cq_normal 632；涉诈 367）
- `human_0729` 的 1447 个 id 沿用 `eval/splits.json` 的冻结拆分；`agent_1022` 与 `human_cq_normal` 按近重复簇整簇分配（簇不跨 split），新正常行按 40/20/40 补入。

## 三个 cohort（重要）

- `human_0729`：1447 条，2026-07-29 人工确认；主门禁与 v1/v2 配对比较只用这个 cohort。
- `agent_1022`：6353 条，来自 `assets/10.22 复核结果.xlsx`，覆盖 2026-10-23..2026-11-04。其中 6246 条采用 `assets/10.22agent和人工复核结果.xlsx` 的人工复核，107 条保留 242 条队列的 spec 复核裁定。风险分布：{'低风险': 6116, '高风险': 176, '合规': 13, '中风险': 48}；涉诈 50 条。
- `human_cq_normal`：1581 条，来自 `assets/重庆行业卡人工复核正常数据.xlsx`，全部为人工复核正常/合规；用于补充正常类，尤其是 gate 的正常样本。
- 全量近重复簇 8586 个（多行簇 91 个，最大 643 条）；新正常行与既有行同簇 0 个，未跨 split。
- 主门禁口径：`split=gate` 且 `cohort=human_0729`（578 条）；`human_cq_normal` gate（632 条）用于正常类 recall；`agent_1022` gate （2540 条）只作外部人工复核对照。

## 标签语义（重要）

新标准把旧的粗标签 `涉诈` 拆成两层：

- `is_violation=1`：命中任一违规/涉诈场景，共 7633 条；其中 4.2 涉诈 919 条，仅 4.1 违规 6714 条；
- `is_fraud=1`：命中 4.2 涉诈场景（均为高风险），共 919 条（human_0729 869 + agent_1022 50 + human_cq_normal 0）；
- `非涉诈` = 合规 + 仅 4.1 违规，共 8462 条。

**二分类任务的目标是 `is_fraud`，不是旧的 `label`。** 旧人工 `label=涉诈` 的 1287 条中，869 条是 4.2 涉诈，418 条降为仅 4.1 违规（负类）。全量涉诈占比 919/9381 = 9.8%。

## 三套任务

v2 的 prompt/rubric 定义在 `eval/task_v2.py`；特征提取与 probe 复评使用
`eval/features.py --task v2`、`eval/probe.py` 与 `eval/report_v2.py`。

- `goldens.jsonl`：`请判断该通话是否属于4.2涉诈场景（而非仅属于4.1违规）：`，选项 ['非涉诈', '涉诈']，全部 9381 条。
- `goldens_scene.jsonl`：`请判断该通话的违规业务大类：`，选项 ['引流第三方平台', '贷款相关', '法律服务', '企业营销与招商服务', '商品推销', '商业地产', '违规催收', '其他']，仅非涉诈的 4.1 违规行（6714 条）。
- `goldens_fraud_type.jsonl`：`请判断该通话的4.2涉诈类型：`，选项 ['个体工商户年报补录收费', '引导贷款用户添加第三方微信', '手机租赁套路贷', '套路运', '网贷平台退息退费', '证券投资类', '贷款相关-ab贷']，仅 `is_fraud=1` 行（919 条）。

## 复核状态

- `needs_review=1`：0 条（human_0729 0 + agent_1022 0 + human_cq_normal 0）。
- 242 条复核队列全部关闭，回填 `label_source=spec_review` 242 条；其中 agent_1022 的 107 条保留 spec 裁定，未被人批复核覆盖。
- 逐条结论与证据见 `runs/labels_v2/review_queue_复核结论.csv`；两种复核来源的 107 条重叠冲突见 `runs/labels_v2/spec_review/conflicts_human1022_vs_spec.csv`。

## v2 probe 复评（2026-11-05）

- 特征：`runs/v2/features_binary.npz`（9381×4×2560，最后一 token，层 8/16/24/32）；probe 选中 layer 24、L2 0.1、class weight balanced。
- dev（选择用，gate 未参与）：bal 0.952、涉诈 recall 0.935、非涉诈 recall 0.969、ECE 0.011。
- 主门禁 `human_0729` gate（578 条）：
  - argmax：bal 0.891、涉诈 recall 0.934、非涉诈 recall 0.848、precision 0.903、PR-AUC 0.976、ECE 0.024（tp/fn/fp/tn = 325/23/35/195）；四项绝对门禁仅涉诈 recall 差 1.6pp，**未通过**。
  - dev 冻结的 `probe@FPR5`（阈值 0.3075）：recall 0.951 / TNR 0.761 / bal 0.856 / ECE 0.029，四项绝对门禁**通过**。
  - dev 冻结的 `probe@FPR10`（阈值 0.1403）：recall 0.983 / TNR 0.626 / bal 0.804 / ECE 0.057，四项绝对门禁**通过**。
  - dev 冻结的 `probe@dev-R95`（阈值 0.390，dev 上 recall≥0.95 前提下最大化 TNR）：recall 0.945 / TNR 0.809 / bal 0.877，仍差 0.5pp **未通过**。
- 采样不确定性：涉诈类 348 条，argmax recall 0.934 的 Wilson 95% CI 0.903–0.956；若真实 recall=0.95，观测到 ≤325/348 的概率约 10.8%。点估计未过门禁，但在 n=348 下无法与 0.95 统计区分；近重复簇使区间偏窄，门禁仍按预注册点估计判定。
- 阈值迁移：dev 选点的 FPR 在 gate 上明显放大——argmax 3.1% → 主门禁负类 15.2% / `human_cq_normal` 14.9%；FPR5 5% → 23.9% / 23.1%；FPR10 10% → 37.4% / 34.2%。原因是负类构成/批次不同（主门禁负类 167/230 为仅 4.1 违规，`human_cq_normal` 为另一批纯合规）。`@FPR5`/`@FPR10` 只是 dev 选点目标，不是线上 FPR 保证，上线前需用目标流量重新校准。
- 补充 cohort（gate，argmax）：`human_cq_normal` 632 条全正常，TNR 0.851（94 条误报）；`agent_1022` 2540 条（19 涉诈）recall 0.684 / TNR 0.992；全量 gate bal 0.938 / recall 0.921 / TNR 0.956。
- 对比 v1 冻结 probe（同 578 条，按 v2 标签重评分）：v1 bal 0.601 / recall 0.902 / TNR 0.300；v2−v1 Δbal +0.290 [0.251, 0.327]，McNemar p=3.03e-29（混入 rubric 变化，仅诊断）。
- 错误：误报以仅 4.1 违规负类为主（正常 109、贷款相关 32），漏报以 4.2 难例为主（引导加微 11、ab贷 9）。
- 完整报告与机读结果：`runs/v2/report.md`、`runs/v2/gate_results.json`；复现命令见报告第 7 节。

## 已知限制

- `agent_1022` 的 6,246 条是 10.22 人工复核，107 条是 spec 复核；两套口径在 107 条重叠上仍有 19 条风险、17 条场景、2 条涉诈分歧，当前按「专项 spec 裁定优先」保留。
- `agent_1022` 只有 50 条 4.2 涉诈，不能单独估计涉诈 recall；它主要补充低/中/高风险违规负类。
- `agent_1022` 与 `human_cq_normal` 都是单批 snapshot；拆分按近重复簇分组而不是时间外推，簇内模板不跨 split，但同一业务的不同模板仍可能跨 split。
- `human_0729` 的 4.1 风险等级/类型多为规则推断，只有 16 条有明确人工风险备注；新标准 22 类涉诈场景中有 15 类在本数据中无样本。
- `human_cq_normal` 的 gate 正常样本为 632 条，全部来自 2026-08 批次，不是时间外推；正常 recall 仍需在真实时间窗上复核。

## 审计字段

- `cohort`：`human_0729` / `agent_1022` / `human_cq_normal`；`source_file`：原始文件。
- `label`：`非涉诈`/`涉诈`（由 `is_fraud` 派生）；`is_violation`/`is_fraud`：0/1。
- `risk_level`：合规/低/中/高；`scene_category`/`scene_subtype`：新分类。
- `label_source`：`old_category`/`manual_note`/`rule:*`/`spec_review`/`human_1022`/`human_cq_normal`。
- `proposed_*`：历史建议字段，复核完成后已清空；原始建议见 `review_queue_combined.csv`。
