# M0.5 评估集（冻结）

- 源文件：`assets/0729复核人工确认数据.csv`（sha256 `0798fa1696625933…`）
- 生成脚本：`eval/prepare.py`（salt `risk-jev-m0.5-v1`）
- 样本：1447 条，全部来自 2026-07-29 单日人工复核数据
- 拆分：cal 580 / dev 289 / gate 578

> 新标准 v2 标签（4.1/4.2 分层、`is_fraud` 二分类目标）见
> `assets/0729复核人工确认数据_v2.csv` 与 `eval_v2/README.md`。
> 当前 v2 数据集 `assets/复核数据_v2.csv` 共 9381 条：
> `human_0729` 1447 + `agent_1022` 6353（10.22 人工复核 6246 + spec 裁定 107）
> + `human_cq_normal` 1581（重庆行业卡正常数据）。评估集以 `eval_v2/` 为准。
> 生成链：`eval/relabel.py` → `eval/merge_review_batch.py`
> → `scripts/backfill_spec_review.py` → `eval/merge_human_review.py`
> → `eval/prepare_v2.py`；诊断脚本 `eval/diagnose_v2.py`。
> 107 条双复核来源冲突见
> `runs/labels_v2/spec_review/conflicts_human1022_vs_spec.csv`
> （当前按「专项 spec 裁定优先」保留）。
> 本目录的 v1 artifacts 保持冻结，不要用 v2 覆盖。

## 标签语义（重要）

源文件 `label` 只有两个值：`正常` 与 `涉诈`。`category` 仅对 `label=涉诈` 的行填写。
复核备注中有 10 条明确写“低风险”、6 条写“高风险”，它们全部仍被标为 `涉诈`。
因此这里的 `涉诈` 应理解为**需要按涉诈/违规处置**的粗标签，不能等同于刑事诈骗；
风险等级（合规/低/中/高）在源数据中没有系统标注，M0.5 不把风险等级作为门禁指标。

`人工复核` 与 `备注` 只覆盖少数行，保留在 goldens 中作为审计线索，不参与评分。

## 拆分

- 按 `(label, 粗类型)` 分层，层内按 `sha256(salt + data_id)` 排序后按 40/20/40 分配。
- `cal`：估计固定 prior、训练线性 probe、拟合温度；`dev`：选层/选正则/选阈值；
  `gate`：只用于最终报告，不参与任何选择。
- 源数据只有一天，没有用户/时间字段，无法做时间外推拆分；这是本评估集的硬限制。
- 已做 SimHash（字符 3-gram, 64-bit）近重复检查：Hamming ≤ 4 的样本对为 0，
  精确重复 content 也为 0；同一天同一话术模板的语义泄漏仍可能存在。

## 二分类任务（goldens.jsonl）

- 问题：`请判断该通话对应的业务是否需要按涉诈风险处置：`
- 选项：['正常', '涉诈']
- 全部 1447 条；`涉诈` 1287 条（89.0%），`正常` 160 条（11.0%）。
- 该任务是不平衡二分类：准确率会被多数类主导，报告必须同时看 `涉诈` recall、
  `正常` recall（TNR）、固定 FPR 下的 recall、macro TPR、PR-AUC 与配对置信区间。

## 类型任务（goldens_category.jsonl）

- 问题：`请判断其涉诈类型：`
- 选项：['贷款相关-ab贷', '贷款相关', '个体工商户年报补录收费', '引导贷款用户添加第三方微信', '证券投资类', '违规催收']
- 只保留 top-6 类型的 1243 条涉诈样本；其余 44 条长尾类型（8 个类型，单项 1–13 条）
  不进入类型任务，但保留在二分类任务中。
- 类型任务的条件准确率不能直接乘二分类准确率当作端到端指标；端到端需另行定义。

## 已知限制

- 单日、单批次数据；同一天的外呼活动可能共享话术，gate 不是时间外推。
- 正常类只有 160 条，gate 约 64 条；`正常` recall 的 95% Wilson CI 约 ±12pp。
- 标签含噪：话术边界依赖规范细节，`正常` 与低风险违规之间可能存在人工分歧。
- `涉诈` 类别内混合了低风险违规与高风险涉诈，二分类标签不携带风险等级。

## 审计字段

- `id`：源 `data_id`；`source_row`：CSV 数据行号（不含表头）；`split`：cal/dev/gate。
- `state`：通话转写原文；`category`：源类型；`category_coarse`：top-6 或 `其他`。
- `risk_note` / `review_note`：源 `人工复核` / `备注`，可为空。

## 产物

- `runs/m0.5/features_binary.npz`：最后一层隐藏状态特征（1447×4×2560），由 `eval/features.py` 生成。
- `runs/m0.5/probe/`：probe 网格、选中头、全量预测（`predictions.jsonl` / `predictions_riskfirst.jsonl`）。
- `runs/m0.5/dev/results.json`：cal/dev 上 raw/L0 各变体的逐行决策与指标。
- `runs/m0.5/selection.json`：cal/dev 上冻结的温度、prior 强度和 probe 配置。
- `runs/m0.5/gate/results.json`：gate 上冻结配置的唯一一次评分（含派生配置）。
- `runs/m0.5/frontier_dev.json` / `frontier_gate.json`：扫阈值得到的操作点前沿（post-hoc）。
- `runs/m0.5/triage.json`：三路分流（自动正常 / 转人工 / 自动涉诈）的 dev 选点与 gate 评估（post-hoc）。
- `runs/m0.5/report.md`：由 `eval/report.py` 渲染的最终报告。
- `assets/0729复核人工确认数据_v2.csv`：0729 人工数据的 v2 重标（`is_violation` / `is_fraud` / `risk_level` / `scene_category`）。
- `assets/复核数据_v2.csv`：**当前 v2 主数据集，9381 条**，三个 cohort：`human_0729` 1447、`agent_1022` 6353、`human_cq_normal` 1581；`cohort` / `source_file` / `label_source` 保留来源。
- `assets/复核数据_v2.pre_human1022.bak.csv`：并入 10.22 人批复核与重庆正常数据之前的 7800 条备份。
- `assets/复核数据_v2.pre_specreview.bak.csv`：242 条专项复核回填之前的备份。
- `runs/labels_v2/summary.json` / `review_queue.csv`：0729 cohort 的标签分布与历史 135 条待复核队列（已完成）。
- `runs/labels_v2/summary_combined.json`：当前 9381 条的标签分布；历史 7800 条版本见 `summary_combined.pre_review.bak.json`。
- `runs/labels_v2/review_queue_combined.csv` / `.xlsx`：**历史** 242 条待复核队列与人工复核工作簿，全部已关闭，保留作审计线索。
- `runs/labels_v2/review_queue_复核结论.csv`：242 条逐条复核结论（`复核_is_fraud` / 场景 / 风险 / 依据 / 来源 / 是否改判）。
- `runs/labels_v2/spec_review/`：复核报告 `复核报告.md`、引擎明细 `verdicts.csv`、合并结论 `final_conclusions.csv`、裁定证据 `disagreements.md`、双复核来源冲突 `conflicts_human1022_vs_spec.csv`。
- `runs/labels_v2/human_review_merge_summary.json`：10.22 人批复核 + 重庆正常数据合并的统计与口径（含 107 条优先级规则）。
- `eval/make_review_workbook.py`：由 `review_queue_combined.csv` 生成复核工作簿（历史工具，可复跑）。
- `eval/merge_human_review.py`：并入 10.22 人批复核与重庆正常数据的合并脚本（默认 dry-run，`--write` 落盘）。
- `runs/labels_v2/extra_splits.json` / `extra_details.jsonl`：agent_1022 + human_cq_normal 的拆分与原始 LLM 标注明细；并入正常数据前的版本见 `extra_splits.pre_normal.bak.json`。
- `runs/labels_v2/probe_v2_diagnostic.json`：v1 probe 在 v2 `is_fraud` 上按 cohort 的重评分诊断（仅 human cohort，非 v2 模型）。
- `eval_v2/`：v2 goldens（二分类 **9381** / 4.1 大类 **6714** / 4.2 类型 **919**）、簇级拆分 `splits.json`、`stats.json` 与 README。
- `eval/task_v2.py`：v2（4.2 涉诈）的 prompt/rubric 定义；`eval/features.py --task v2` 据此提特征。
- `runs/v2/features_binary.npz`：v2 全量特征（9381×4×2560，最后一 token，层 8/16/24/32）。
- `runs/v2/probe/`：v2 probe 网格、选中头与全量预测（`predictions.jsonl`）。
- `runs/v2/report.md` / `runs/v2/gate_results.json`：由 `eval/report_v2.py` 生成的 v2 probe 主门禁、FPR5/FPR10 操作点、与 v1 配对比较及错误分析。
- `runs/external_cq/`：重庆行业卡外部批次汇总（`summary_by_file.csv` / `summary_scene_long.csv`）、去重后的外部正常集 `normal_1030.jsonl`、时间外推池 `forward_1103_1113.jsonl`、冻结 probe 打分结果（`normal_1030/`、`normal_1030_unmatch/`、`forward_1103_1113/`）、外部复核工作簿 `review_queue_external.xlsx`、抽样复核 `review_sample.xlsx`、已确认正例 `confirmed_positive.csv`、抽样计划 `sampling_plan.csv`、标签策略 `labeling_policy.md` 与复评报告 `report.md`；打分脚本为 `eval/score_external.py`。
- `eval/external_labels.py`：解析外部 `CQ_Agent_res`（违规/涉诈 + 风险 + 场景/子类），并生成仅供复核参考的 v2 建议标签。
- `eval/make_external_review_workbook.py`：由外部 JSONL 与冻结 probe 预测生成复核工作簿（agent 涉诈 / 正常冲突 / 旧涉诈-agent 正常三个队列）。
- `eval/plan_external_review.py`：生成已确认正例、按场景分层抽样计划（88/193）与抽样汇总。
- `eval/make_external_sample_workbook.py`：把抽样清单渲染成人工复核工作簿。
- `eval/apply_external_sample.py`：读取填好的抽样复核结论，按场景估计 precision/Wilson CI，并按「点估计≥0.90 且下界≥0.70」自动采纳该场景剩余行，产出 `labels_external_final.csv`。
