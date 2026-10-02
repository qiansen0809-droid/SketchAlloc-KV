# LU-KV 跨场景失效：第一轮诊断（非新算法）

> 2026-10-02。本文件记录**已发表结果中的失效线索**与下一步可证伪测试。不能把论文表格中的落差直接归因为静态预算画像，也不能把小样本 proxy 测试当作答案质量实验。本工作不等于 ResearchPilot 阶段 C 的完整 Part 3。

## 已找到的具体线索

LU-KV 原文的 LongBench 明细表在 80% 压缩、SnapKV token scorer 下出现同类任务内部的反向结果（表 5；分数越高越好）：

| 模型 | 数据集 | Full KV | 均匀预算 | LU-KV | LU − 均匀 |
|---|---|---:|---:|---:|---:|
| Mistral-7B-v0.3 | LCC 代码补全 | 65.60 | 63.37 | 53.76 | −9.61 |
| Mistral-7B-v0.3 | RepoBench-P 代码补全 | 60.92 | 58.59 | 61.62 | +3.03 |
| Llama-3.1-8B | LCC 代码补全 | 63.43 | 66.52 | 59.64 | −6.88 |
| Llama-3.1-8B | RepoBench-P 代码补全 | 52.59 | 54.26 | 59.27 | +5.01 |

出处：[LU-KV 正式 ICML 2026 论文](https://proceedings.mlr.press/v306/tang26i.html)，[可检索的原文表 5](https://arxiv.org/html/2602.08585)。仓库评测代码对 LCC 与 RepoBench-P 使用同一个 `code_sim_score`，所以不是两个完全不同分数量纲的比较。50% 压缩时 LCC 的上述大幅落差并不存在；Qwen2.5-32B 也没有出现相同方向的落差。因此首要问题是 **Llama/Mistral + 强压缩 + LCC 的特定失效**，不是“所有代码任务都使 LU-KV 失效”。表中个别均匀预算分数高于 Full KV，说明还须警惕生成/评测波动。

LU-KV 附录 F.2 仅比较两种偏阅读理解的离线标定文本在各任务上的迁移，没有给出 LCC 专用与 RepoBench-P 专用画像的交叉矩阵。这个空缺值得测，但“给任务各训练一张表”本身已接近 Task-KV、DynamicKV、EntroKV，不自动构成论文创新。[LU-KV 附录 F.2](https://arxiv.org/html/2602.08585)、[DynamicKV](https://aclanthology.org/2025.findings-emnlp.426/)、[EntroKV](https://proceedings.mlr.press/v306/gao26a.html)。Tangram 的多任务观察则提供了相反证据：头保留排序可能相当稳定；它用的不是 LU-KV 的长程效用曲线，因此这里只作为需要正面检验的反证。[Tangram 预印本](https://arxiv.org/html/2606.06302v2)。

## 先区分四种解释

1. **预算迁移失效**：小说标定得到的 head 预算确实不适合 LCC；独立 LCC 上下文标定的预算在新 LCC 上能明显降低 LU oracle proxy loss，并最终改善真实代码质量。
2. **头内 scorer 失效**：SnapKV 对 LCC 关键 token 排序错误；无论预算怎样分，增益都有限。若 task profile 的 oracle proxy 也不改善，优先怀疑此项。
3. **长度/格式混杂**：LCC 与 RepoBench-P 的 prompt 长度、代码前缀与问题格式不同。首轮只选同一 token 长度区间的独立 context；不截断原文。
4. **生成与评测波动**：小的代码补全分数差可能由采样、评测或样本组成造成。首轮只找信号；真正确认需要成对的下游质量评估与更大样本。

## Gate −1：只寻找预算迁移信号

准备脚本从 LongBench 的 LCC、RepoBench-P、PassageRetrieval-en、MultiNews 各取若干**不同上下文**，按上下文分成标定与留出。它沿用本仓库的 LU-KV full-attention oracle 收集代码、SnapKV scorer 和同一分配求解器。分析脚本构建四类画像：仓库自带 LU 静态画像、所有场景合并标定的画像、各任务画像、均匀预算；在完全留出的上下文上做交叉测试，并审计实际保留 KV 条目数。

这一轮只看两个问题：LCC 专用画像在新 LCC 上是否稳定优于静态/混合画像？它在 RepoBench-P 上是否反而变差？若两问都不成立，**停止“跨代码子场景预算转移失效”假设**，不要再加复杂路由器。若成立，也只是候选现象；必须用真实生成质量复核，并排除 scorer 与 prompt 长度混杂。

结果还提供两个**事后诊断**，都使用留出样本的未来 oracle，不能上线：`fixed_scorer_budget_headroom` 是在保持 SnapKV token 排序时，使用该 context 自己的 LU 求解预算可减少多少代理损失；由于 LU 求解有凸包近似、总条目数也可能差几个，它只是候选预算的差值，**不是严格最优上界**，可以为负。`token_scorer_headroom_at_official_budget` 是保持官方 LU 头预算，但用 oracle 直接选择 token 时可减少多少代理损失；它才是固定预算下的头内 token 选择上界。若前者很小、后者很大，继续研究头预算很可能抓错了瓶颈。比较前还要核对实际 KV 条目数。

数据来自 LongBench 官方 test split，内部按 context 划分仅适合探索，不是可投稿的独立最终测试集；正式论文必须另留未参与假设选择的外部测试。[LongBench 官方仓库](https://github.com/THUDM/LongBench)。

## 服务器执行（Llama-3.1-8B，48 GB）

使用新目录，避免仓库旧的 step-1 采集脚本覆盖同名 `context_*` 输出。以下 4 个场景 × 每场景 4 个 context 是最小 smoke test（2 标定 + 2 留出），**不可据此宣布统计显著**；若出现信号，再将 `--per-task` 提至 10 以上并增加独立测试。

```bash
cd /root/autodl-tmp
git clone --branch codex/cross-scene-failure-gate https://github.com/qiansen0809-droid/SketchAlloc-KV.git CrossScene-Gate
cd CrossScene-Gate
python -m pip install -e . --no-deps
python -m pytest -q tests/test_cross_scene_gate.py

python -m evaluation.cross_scene.prepare \
  --model-path /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct \
  --out-dir results/cross_scene/selection_v1 \
  --per-task 4 --calibration-per-task 2 \
  --min-tokens 2048 --max-tokens 4096

python evaluation/curve_data/step1_llama.py \
  --model_path /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct \
  --dataset_path results/cross_scene/selection_v1/profile_samples.jsonl \
  --output_dir results/cross_scene/raw_v1 \
  --cuda_device 0 --max_new_tokens 64 --methods snapkv

python -m evaluation.cross_scene.analyze \
  --manifest results/cross_scene/selection_v1/manifest.json \
  --raw-root results/cross_scene/raw_v1 \
  --official-profile evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy \
  --output-dir results/cross_scene/analysis_v1 \
  --method snapkv --sink 4 --window 32 --compression 0.80
```

若准备数据时报某任务在 2048–4096 tokens 内样本不足，先记录报错，再将 `--min-tokens` 调为 1024 或 `--max-tokens` 调为 6144；不要悄悄截断。后续分析仍须按长度分层检查。第一次跑完请发回 `results/cross_scene/analysis_v1/transfer_results.json` 与 `per_context.csv`。`results/` 中的原始上下文和数组不应推到 GitHub。

## 判读边界

`mean_proxy_loss` 越低越好；`task_profile_vs_official_mean_proxy_loss_reduction` 为正表示本任务画像优于官方 LU 画像的代理损失。仅有画像数值差、仅有一个 context 获益、或缓存预算不相等，都**不算**发现失效。即使代理结果通过，也须在相同留出样本上用同样总预算、同样 SnapKV scorer 比较真实代码补全/检索/摘要质量，并单独设立最终未触碰的测试集。
