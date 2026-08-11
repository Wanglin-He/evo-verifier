# evo-verifier

三个人分头写 A1–A6 / B7–B10 / B11–B14 的检测器，这个包是三份代码共用的地基：
**项定义、人类标签、输出契约、对齐评测**。检测器本身不放在这里。

零第三方依赖，Python 3.11+，可以整个目录搬到任何仓库。

```bash
uv run --no-project --with pytest python -m pytest -q          # 35 passed
uv run --no-project python -m evo_verifier --group B7-B10 stats
uv run --no-project python -m evo_verifier --group B7-B10 evaluate <reports_dir>
```

## 模块

| 文件 | 作用 |
|---|---|
| `items.py` | 14 项的冻结定义：项号、英文 key、CSV 列名、族、负责人分组、是否需要仿真器 |
| `labels.py` | 读标注平台 `/api/export` 的 CSV → 类型化标签，任何不认识的值直接报错 |
| `report.py` | `report.json` 的结构与校验、族分数与 `Score_full` 聚合 |
| `evaluate.py` | 逐项 TP/FP/FN/TN、precision/recall/F1、平衡准确率、Cohen's κ、覆盖率 |
| `data/annotations-2026-08-11.csv` | 标注导出的冻结快照（607 条，2026-08-11 下载） |

## 已经敲定的契约

**① 项 ↔ CSV 列名**：`COLUMNS` 是唯一锚点。中文列名是标注员实际回答过的问题，
英文 key 只是代码里的别名，改 key 不改语义。

**② 标签编码**：`满足→PASS`、`不满足→FAIL`、`不涉及→NA`、**空→MISSING**。

空不是通过。334 条旧版转换记录的 A4/A5/A6/B9/B13/B14 六列是空的，
把空读成"满足"会让 B9 的分母从 272 涨到 606。`test_labels.py` 盯着这一条。

**③ 单项结果的字段**：`score / prediction / threshold / confidence / coverage /
tools / raw_measurements / failure_reason / repair_hint`。校验在 `ItemResult.__post_init__`
里强制执行，写错直接抛 `ReportError`：

- `coverage` 是 `not-applicable` 或 `unsupported` → 必须没有 `score`；
- 有 `score` → 必须有 `prediction`，且 `score ∈ [0,1]`；
- `prediction == fail` → 必须写 `failure_reason`；
- `confidence < 0.50` → 自动变成 `abstain`，不硬猜。

**④ N/A 与 abstain 是两种状态**：`not-applicable` = 契约没这个要求（离开聚合，
分子分母同时剔除）；`abstain` = 工具答不了（不算错，计入覆盖率损失）。

**⑤ 聚合**：`Score_full = 100·Σ w_f·S_f / Σ w_f`，族权重 `{semantic .20, static .25,
motion .30, physics .25}`，只对有证据的族求和。族权重和公式属于冻结项，永不调整。

**⑥ 阈值**：`DEFAULT_THRESHOLD = 0.70` 是协议占位值，不是设计值。
每项的 τ 要在开发集上校准后冻结。

**⑦ 评测口径**：positive = FAIL。人类 MISSING/NA 出表；验证器 abstain / 未上报出表，
但都以计数形式出现在 `abstained` / `not_reported` / `verifier_na` 里。

## 还没敲定的（需要三个人一起定）

- **族归属**：提案里 B10/B12/B13 是多对多，A4 根本没归族。`items.py` 里现在
  每项给了一个主族，这是我们的选择不是协议的，见 `FAMILY_ASSIGNMENT_NOTE`。
- **归一化常量**：`D`（整体包围盒对角线，B10 用）、`Δq_j`（关节行程）、
  `L_j`（活动连杆对角线，B12 用）。B10 的分数完全取决于 D 的定义方式，
  必须只有一份实现。
- **FK 采样约定**：32 步采样时其他关节停在哪、区间取声明 limit 还是几何可用范围。
  B9/B10/B11/B13 全都依赖它。
- **契约（contract）格式**：B7 判"父子关系是否正确"必须先知道预期的父子关系。
  这份契约从 prompt 抽取，建议每条带 `source: explicit | prior` 和 `confidence`。

## 数据现状

607 条记录（273 条按 14 项新 schema 标满，334 条旧版转换）。各项失败率差异极大：

| 项 | 人类判失败 | 有效样本 |
|---|---|---|
| B7 父子关系 | 8 | 605 |
| B8 关节存在性 | 16 | 606 |
| B9 关节类型 | 13 | 272 |
| B10 位置与轴向 | 10 | 584 |

B7–B10 一共 47 个正例。直接后果：**逐项 F1 在这个量级上是噪声，阈值没法在真实
数据上校准**。一个"永远预测通过"的基线在 B7 上准确率 98.7%，但 κ = 0：

```
 item     n+    TP    FP    FN     TN    prec     rec      F1   bal-acc    kappa    cover
   B7      8     0     0     8    597       -   0.000       -     0.500    0.000    1.000
```

所以 κ 和 `n+` 列必须一起看，只报准确率会骗人。

计划中的绕法是**合成失败注入**：从标注为满足的资产上程序化地制造 B7/B8/B9/B10
的失败（改 parent、删关节、换类型、平移原点、旋转轴向），用带强度梯度的合成正例
定阈值，真实的 47 个正例留作一次性检验。
