# LU-KV 泛化压力测试

> 目的：先判断 LU-KV 的静态 cross-head profile 在 workload shift 下是否存在稳定、可复现的失配，再决定是否把 SketchAlloc-KV 的主叙事转为“低成本适应 profile shift”。

本测试不预设 LU-KV 泛化差。若 source profile 与 target-calibrated profile 的差异很小，则该方向应停止。

## 1. 三个主维度

### A. Domain / task-family shift

固定模型、scorer、压缩率和总预算，只改变测试任务族。LongBench 按以下 family 统计：

- single-doc QA: narrativeqa, qasper, multifieldqa_en
- multi-doc QA: hotpotqa, 2wikimqa, musique
- summarization: gov_report, qmsum, multi_news
- few-shot: trec, triviaqa, samsum
- synthetic: passage_count, passage_retrieval_en
- code: lcc, repobench-p

screening 模式每个 family 先跑一个代表任务；full 模式跑全部 16 个任务。

### B. Context-length shift

固定同一静态 profile，在 RULER 上比较 8K / 16K / 32K；full 模式额外包含 64K。

核心问题不是“长度越长分数是否下降”，而是同一 profile 相对重新 calibration 的 profile 是否出现越来越大的 regret。

### C. Compression shift

固定任务与 profile，比较 50% / 80% / 90% compression。重点观察高压缩下 domain/length shift 是否被放大。

## 2. 关键比较：source-profile vs target-profile

只看一个 source profile 的绝对分数，不能证明“泛化失败”。推荐定义：

```
Generalization Regret(target)
  = Score(target-calibrated profile on target)
  - Score(source profile on target)
```

因此实验分两层：

1. **Screening**：先把当前 source profile 扔到不同 domain / length / compression 上，看是否存在明显脆弱区域。
2. **Confirmatory**：对脆弱 target 构建独立 target-calibrated LU profile，在完全相同的 scorer、模型、压缩率、评测集上比较 source vs target profile。

target profiling 数据必须与 target test 样本隔离，禁止直接用测试集构建 profile。

## 3. SketchAlloc 的公平比较

若后续测试 SketchAlloc：

- LU-KV 与 SketchAlloc 必须使用相同 source calibration 信息；
- SketchAlloc 在 target 上只能读取部署时允许的无答案 probe；
- 不允许用 target gold answer 或 target test label 更新方法；
- target-calibrated LU 只作为诊断上界，不作为同信息量 baseline。

最有解释力的结果形态是：

```
LU-source < SketchAlloc-source ≈ LU-target
```

这表示性能损失主要来自 profile shift，而 SketchAlloc 在不读取 target label 的情况下恢复了其中一部分。

## 4. 执行工具

生成/运行矩阵：

```bash
python -m evaluation.sketchalloc.generalization_matrix \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --profile source=/path/to/lu_curve.npy \
  --longbench-path /path/to/longbench \
  --ruler-path /path/to/ruler \
  --dimension all \
  --mode screening \
  --device cuda:0 \
  --output-root results/generalization
```

默认只生成计划，不执行模型。确认计划后加：

```bash
--execute
```

汇总：

```bash
python -m evaluation.sketchalloc.summarize_generalization \
  --manifest results/generalization/run_manifest.jsonl \
  --output-csv results/generalization/summary.csv \
  --output-json results/generalization/summary.json
```

## 5. Go / No-Go

**Go：** 在至少两个独立 shift 轴上出现稳定 source→target regret，且重新 target calibration 能显著恢复性能；最好高压缩下问题更明显。

**No-Go：** source profile 在不同 domain / length / compression 下都与 target-calibrated profile 接近，或差异只来自少量异常任务、随机种子或评测噪声。

不要用“某个任务分数低”直接宣称泛化失败。必须把“压缩本身的难度”和“静态 profile 失配”分开。
