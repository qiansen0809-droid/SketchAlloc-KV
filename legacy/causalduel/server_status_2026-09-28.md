# CausalDuel-KV 服务器与项目现状报告

**时间：2026-09-28**  
**服务器环境：AutoDL**  
**项目目录：** `/root/autodl-tmp/CausalDuel-KV`  
**开发分支：** `gate0-poc`

---

## 1. 当前项目状态总览

目前项目已经完成了 Gate-0 与 Gate-0-v2 的核心实验链路，代码、模型、数据、LU marginal profile、MiniGate 数据集以及两轮正式 24-prompt 实验均已落盘。

当前研究阶段可以概括为：

```text
研究问题定义                    ✅
LU baseline 接入                 ✅
局部 budget-preserving swap      ✅
candidate generation             ✅
正式 calibration                 ✅
固定 MiniGate discovery set      ✅
Query-tail Gate-0                ✅
Answer-onset Gate-0-v2           ✅
正式 24-prompt 实验              ✅
实验结果 summary                 ✅

最终 prompt-conditioned scorer   ❌
fresh confirmatory set           ❌
physical KV pruning backend      ❌
16K / 32K 正式实验               ❌
多模型验证                       ❌
TTFT / throughput / peak VRAM    ❌
最终论文实验                     ❌
```

当前最重要的研究结论是：

> LU 的静态预算存在明显的 prompt-specific 局部改进空间，但目前基于 FullKV 行为相似度的 selector 无法稳定识别 beneficial swap。

---

# 2. Git 状态

当前分支：

```text
gate0-poc
```

当前 commit：

```text
5d43362e26829cb4c3e89c40846afe19dcd98862
```

最近 5 个提交：

```text
5d43362  Document post-Gate0 answer-onset discovery protocol
93123d2  Add answer-onset Gate-0-v2 summary
f57e018  Add answer-onset Gate-0-v2 discovery runner
3091e40  Add future-facing FullKV answer trace helpers
1d8eab2  Clamp probability overlap to its mathematical range
```

说明当前本地仓库已经同步到了 answer-onset discovery protocol 的最新实现。

---

# 3. 磁盘状态

系统盘：

```text
Size: 30G
Used: 849M
Avail: 30G
Use: 3%
Mount: /
```

数据盘：

```text
Size: 100G
Used: 31G
Avail: 70G
Use: 31%
Mount: /root/autodl-tmp
```

结论：

- 系统盘空间非常充足；
- 数据盘仍有约 70GB 可用；
- 当前实验继续运行不存在明显磁盘压力。

---

# 4. 模型状态

模型目录：

```text
/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct
```

占用空间：

```text
30G
```

状态：

```text
✅ 模型已完整下载
✅ 可正常加载
✅ 已用于 LU profile、MiniGate、Gate-0 和 Gate-0-v2
✅ 多次 8K 实验已成功完成
```

因此当前模型文件无需重新下载。

---

# 5. 数据集状态

## 5.1 本地数据目录

```text
/root/autodl-tmp/datasets
```

总占用：

```text
130M
```

---

## 5.2 RULER 8192

本地文件：

```text
/root/autodl-tmp/datasets/ruler_8192/test-00000-of-00001.parquet
```

当前显示大小：

```text
65M
```

状态：

```text
✅ 已完整下载
✅ 已成功生成 test split
✅ 已用于 MiniGate
```

---

## 5.3 LongBench

已经成功下载并使用过以下任务：

```text
narrativeqa
qasper
multifieldqa_en
hotpotqa
2wikimqa
musique
```

这些数据目前主要位于 Hugging Face datasets cache 中。

状态：

```text
✅ 已下载
✅ 已成功用于 MiniGate
⚠ 当前尚未单独盘点其 cache 绝对路径
```

---

## 5.4 SQuALITY

正式 calibration 数据已生成：

```text
results/gate0/profile_data/squality_train_6x5.jsonl
```

文件大小：

```text
167494 bytes
```

状态：

```text
✅ 已准备
✅ 已用于 30-pair LU marginal profile
```

---

# 6. LU Profile 文件

正式 LU profile：

```text
results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32.npy
```

大小：

```text
202880 bytes
```

正式 marginal profile：

```text
results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz
```

大小：

```text
389504 bytes
```

状态：

```text
✅ 已完成
✅ 已重复用于 Gate-0 和 Gate-0-v2
✅ 是后续实验必须保留的核心文件
```

---

# 7. LU Profile Raw 数据

当前保存了 6 个 calibration context：

```text
results/gate0/lu_profile_raw/context_0
results/gate0/lu_profile_raw/context_1
results/gate0/lu_profile_raw/context_2
results/gate0/lu_profile_raw/context_3
results/gate0/lu_profile_raw/context_4
results/gate0/lu_profile_raw/context_5
```

每个 context 都包含：

```text
snapkv.npy
question_0.npy
question_1.npy
question_2.npy
question_3.npy
question_4.npy
```

即：

```text
6 contexts × 5 questions = 30 question-level calibration files
```

这些 raw 文件单个约 5.9MB ～ 6.75MB。

状态：

```text
✅ 原始 profile 数据完整保留
```

这部分对于后续重新计算 marginal、检查 profile、做 attribution 分析都很有价值。

---

# 8. MiniGate 数据

固定 MiniGate：

```text
results/gate0/minigate/minigate_24.jsonl
```

文件大小：

```text
835886 bytes
```

当前数据结构：

```text
24 prompts
24 unique contexts
```

组成：

```text
8 × RULER retrieval
8 × LongBench single-document QA
8 × LongBench multi-document QA
```

状态：

```text
✅ 已固定
✅ 已用于正式 Gate-0
✅ 已用于正式 Gate-0-v2
```

---

# 9. 正式实验完成情况

## 9.1 Query-Tail Gate-0

正式输出目录：

```text
results/gate0/minigate/formal24
```

JSON 数量：

```text
24
```

说明：

```text
✅ 24 / 24 prompts 全部完成
```

---

## 9.2 Answer-Onset Gate-0-v2

正式 discovery 目录：

```text
results/gate0/answer_onset/discovery24
```

JSON 数量：

```text
24
```

说明：

```text
✅ 24 / 24 prompts 全部完成
✅ 6 candidates / prompt
✅ 共 144 candidate pairs
```

---

# 10. Answer-Onset Discovery24 结果

## Trace = 1

```text
pairs/prompts = 144 / 24

Overlap sign   = 41.67%
KL sign        = 41.67%
JS sign        = 43.75%
Top10 sign     = 4.86%
PseudoNLL sign = 47.22%
LU sign        = 50.00%

Overlap - LU   = -8.33 percentage points
```

Bootstrap：

```text
mean = -0.08154
95% CI = [-0.19444, 0.03472]
```

Selector：

```text
fallback        = 4.17%
beneficial exists = 79.17%
improve         = 37.50%
harm            = 58.33%
mean gain       = -0.004366
oracle gain     = +0.014144
```

---

## Trace = 4

```text
Overlap sign   = 46.53%
KL sign        = 47.22%
JS sign        = 47.22%
Top10 sign     = 17.36%
PseudoNLL sign = 36.81%
LU sign        = 50.00%
```

Selector：

```text
fallback        = 12.50%
beneficial exists = 79.17%
improve         = 20.83%
harm            = 66.67%
mean gain       = -0.007569
oracle gain     = +0.014144
```

---

## Trace = 8

```text
Overlap sign   = 48.61%
KL sign        = 50.69%
JS sign        = 49.31%
Top10 sign     = 22.22%
PseudoNLL sign = 43.06%
LU sign        = 50.00%
```

Selector：

```text
fallback        = 4.17%
beneficial exists = 79.17%
improve         = 29.17%
harm            = 66.67%
mean gain       = -0.009148
oracle gain     = +0.014144
```

---

# 11. 当前结果的研究结论

两轮实验都表明：

```text
Query-tail FullKV similarity    ❌
Answer-onset FullKV similarity  ❌
```

目前都不能稳定预测真实 swap utility。

但是：

```text
beneficial swap exists = 79.17%
```

说明：

> LU 静态预算并不是多数 prompt 的局部最优。

因此真正值得继续研究的是：

```text
prompt-conditioned marginal utility
```

而不是继续优化：

```text
FullKV behavioral similarity
```

---

# 12. 当前代码完成度

目前代码已经覆盖：

```text
✅ LU keep-count override
✅ budget-preserving swap
✅ donor / receiver candidate generation
✅ LU marginal profile
✅ MiniGate generation
✅ RULER local parquet support
✅ AnswerNLL evaluation
✅ task score
✅ query-tail probe
✅ answer-onset probe
✅ pseudo-answer trace
✅ KL / JS / Overlap / TopK / NLL metrics
✅ summary
✅ cluster bootstrap
✅ candidate tests
✅ marginal tests
✅ metric tests
```

目前代码已经是一个较完整的研究实验框架，而不只是 smoke test。

---

# 13. 尚未完成的代码与实验

目前还缺：

```text
❌ Prompt-conditioned marginal utility scorer
❌ Oracle attribution feature extractor
❌ Attention/value contribution analysis
❌ fresh held-out confirmatory set
❌ 16K / 32K formal benchmark
❌ 多模型验证
❌ physical KV pruning
❌ peak VRAM evaluation
❌ TTFT
❌ throughput
❌ decode latency
```

---

# 14. GPU 当前状态

当前 GPU：

```text
NVIDIA vGPU-48GB
Total memory: 49140 MiB
```

当前状态：

```text
Memory-Usage: 0 MiB
GPU-Util: 0%
Temperature: 41C
```

进程：

```text
No running processes found
```

结论：

```text
✅ 当前没有 GPU 实验在运行
✅ GPU 已完全释放
✅ 可以安全关闭实例或启动下一阶段实验
```

---

# 15. 当前结果目录大小

整个 Gate-0 结果目录：

```text
results/gate0
```

当前占用：

```text
205M
```

相对于 30GB 模型来说非常小。

因此：

```text
实验结果建议全部保留
```

没有必要为了空间删除这些结果。

---

# 16. 强烈建议备份的文件

## P0：必须备份

```text
results/gate0/lu_profile/
gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz
```

```text
results/gate0/minigate/
minigate_24.jsonl
minigate_24_manifest.json
```

```text
results/gate0/minigate/formal24/
```

```text
results/gate0/minigate/formal24_summary/
```

```text
results/gate0/answer_onset/discovery24/
```

```text
results/gate0/answer_onset/discovery24_summary.json
```

---

## P1：建议备份

```text
results/gate0/profile_data/
```

```text
results/gate0/lu_profile_raw/
```

```text
/root/autodl-tmp/datasets/ruler_8192/
test-00000-of-00001.parquet
```

---

# 17. 当前服务器状态结论

目前服务器状态非常健康：

```text
系统盘使用：3%
数据盘使用：31%
GPU：空闲
模型：完整
RULER：完整
LongBench：已缓存
SQuALITY：已准备
LU profile：完整
MiniGate：完整
Gate-0 formal24：24/24
Gate-0-v2 discovery24：24/24
```

当前没有任何必须立即重新下载或重跑的关键文件。

---

# 18. 推荐下一步

当前不建议直接扩大到 16K/32K。

更合理的下一步是：

```text
P0  冻结当前 Gate-0 结果
P1  备份结果
P2  对 144 candidate pairs 做 Oracle Attribution Analysis
P3  提取 attention / value / head contribution 特征
P4  分析 feature 与真实 AnswerNLL gain 的关系
P5  设计 Prompt-Conditioned Marginal Utility scorer
P6  当前 24 prompts 仅作为 discovery set
P7  冻结新 scorer
P8  新抽 fresh non-overlapping confirmatory set
P9  通过后再做 16K / 32K、多模型和 physical KV backend
```

---

# 19. 当前项目一句话总结

> **工程链路已经基本搭建完成，当前服务器上的模型、数据、LU profile 和两轮 24-prompt 正式实验均已落盘；真正需要解决的已不是环境与工程问题，而是如何从 FullKV behavioral similarity 转向更直接的 prompt-conditioned marginal utility estimation。**
