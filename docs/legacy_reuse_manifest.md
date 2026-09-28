# CausalDuel-KV 遗产复用清单

## 结论

旧项目不是整体报废。它的 selector 已经失败，但“固定预算动作生成 + 真实收益评估”的实验底座恰好是 SketchAlloc-KV 的起点。

## 原样保留

| 资产 | 用途 |
|---|---|
| `kvpress/presses/LU_press.py` | exact keep-count override 与 LU baseline |
| `kvpress/attention_patch.py` | 继承逻辑 head masking；仅用于算法 Gate 0 |
| `evaluation/curve_data/` | LU profile 与 marginal curve 生成 |
| `evaluation/sketchalloc/build_candidates.py` | budget-preserving swap、动作合法性、预算守恒 |
| `evaluation/sketchalloc/metrics.py` | answer NLL 与任务指标基础函数 |
| `evaluation/sketchalloc/prepare_minigate.py` | 固定数据抽样与 manifest |
| `evaluation/sketchalloc/prepare_squality_profile.py` | 独立 LU profile 数据准备 |
| 原有 candidate/marginal/metric tests | 防止预算与评价链路回归 |

## 需要重写

| 资产 | 改造方向 |
|---|---|
| `minigate_runtime.py` | 保留 cache clone、answer NLL；移除 behavior selector 语义 |
| `summarize_*` | 改为 action matrix、best-fixed、family-static、cross-over 和低秩统计 |
| candidate policy | 从每 prompt 六个临时候选改为跨 prompt 固定动作字典 |
| result schema | 增加稳定 `action_id`、split、document_id、task_family、物理预算信息 |

## 只归档

以下文件已复制到 `legacy/causalduel/`，不得从该目录重新接回主方法：

- `suffix_replay.py`
- `run_answer_onset.py`
- `summarize_answer_onset.py`
- `run_minigate.py`
- `summarize_minigate.py`
- 2026-09-28 服务器状态报告

它们可用于复现“行为相似度 selector 失败”的负结果。

## 服务器结果迁移

从 `/root/autodl-tmp/CausalDuel-KV` 复制下列路径到新仓库的 `results/legacy_causalduel/`：

```text
results/gate0/lu_profile/
results/gate0/lu_profile_raw/
results/gate0/profile_data/
results/gate0/minigate/minigate_24.jsonl
results/gate0/minigate/minigate_24_manifest.json
results/gate0/minigate/formal24/
results/gate0/answer_onset/discovery24/
results/gate0/answer_onset/discovery24_summary.json
```

迁移后先生成 SHA256 清单，再进行任何格式转换。原始 JSON/NPY/NPZ 保持只读；新分析只写入 `results/sketchalloc/`。

## 许可证与来源

本仓库保留上游 Apache-2.0 `LICENSE` 和 `CITATION.cff`。论文和 README 必须明确区分：

- KVPress 上游代码；
- LU-KV 作者实现或依据论文的复现；
- CausalDuel-KV 分支新增代码；
- SketchAlloc-KV 新增代码。

未经验证不得把第三方复现称为官方实现。
