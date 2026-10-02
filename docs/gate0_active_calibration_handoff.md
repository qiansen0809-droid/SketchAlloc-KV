# Active Calibration Gate 0 — 服务器交接说明

> 日期：2026-10-02。这里是**新方向的轻量先验实验**，不是旧 SketchAlloc-KV 的 Gate 0，也不是 ResearchPilot 正式阶段 C/E 的完整验收。

## 一句话说明

先检验 LU-KV 的离线头预算画像是否真的依赖“选了哪几个标定问题”：同样的 SnapKV token 排序和 80% 压缩率，只改变参与平均的标定问题，在未参与分配的留出问题上看预算和 LU oracle 代理损失是否变化。若连这个空间都没有，就不要急着研发复杂的主动选题算法。

## 复用与新增

- 复用本仓库 `evaluation/curve_data/step1_llama.py` 生成的 `context_*/question_*.npy`（逐问题长程 oracle utility）和 `snapkv.npy`（固定 token scorer）。也复用 `step2_compute_curve.py` 的 LU-KV 全局头分配与运行时预算取整逻辑；不改压缩后端。
- 新增 `evaluation/active_calibration/gate0.py`。它将问题随机拆成 20 个候选标定问题和 10 个完全留出的代理评估问题，在 20 个中随机抽 4/8/16 个，重复 20 次。所有方法都用同一压缩率、同一 scorer、同一留出集合。
- `full_calibration` 是 20 个候选问题的平均画像，不是真实最优策略；`uniform_budget` 是每头同样压缩率的参照。输出 `Δproxy_loss = subset_loss - full_calibration_loss`，正数表示少量标定更差。
- 这是**离线重放**：所有 30 个问题最终都需要先收集 full-attention 曲线，但“选子集”的模拟不会看留出问题的曲线。收集所有曲线是为了在离线评估时知道答案，不代表主动方法上线时要付 30 次标定成本。

## 已知服务器环境

环境包记录的是 Ubuntu 22.04、单卡 RTX 4090（约 48GB 显存）、Python 3.12.3、PyTorch 2.5.1+cu124、Transformers 4.53.3、NumPy 2.1.3；`kvpress` 已在旧 `/root/autodl-tmp/SketchAlloc-KV` 以 editable 模式安装。环境包没有模型权重路径、原始曲线目录或数据挂载路径，下面用 `<MODEL_PATH>` 表示你服务器上的实际 8B 模型目录。脚本不需要 scikit-learn、flash-attn 或 xformers。

## 运行

若当前服务器分支有未提交工作，最安全的是**另克隆到新目录**，不要强行切换或覆盖原 checkout：

```bash
cd /root/autodl-tmp
git clone --branch codex/active-calibration-gate0 https://github.com/qiansen0809-droid/SketchAlloc-KV.git ActiveCalibration-Gate0
cd ActiveCalibration-Gate0
python -m pip install -e . --no-deps
python -m pytest -q tests/test_active_calibration_gate0.py
```

如果已经有这次实验的 `context_*/question_*.npy` 和相应 `snapkv.npy`，直接跳到分析命令。**只有静态 `*_avg_ratio.npy` 不够**。否则先用仓库自带的 30 问题小说样本收集原始曲线：

```bash
python evaluation/curve_data/step1_llama.py \
  --model_path <MODEL_PATH> \
  --dataset_path evaluation/curve_data/novel.jsonl \
  --output_dir results/active_calibration/raw_novel_64 \
  --cuda_device 0 \
  --max_new_tokens 64 \
  --methods snapkv
```

`step1_llama.py` 会重建其输出目录下同名 `context_0` 等子目录，务必使用**新建且未存放重要结果**的 `--output_dir`；不要指向旧实验目录。显存不足或只想检查流程时，可另用一个新目录并把 `--max_new_tokens` 改为 `16`，但 16-token 结果不能与 64-token 结果混为同一实验。

然后运行只用 CPU/NumPy 的分析：

```bash
python -m evaluation.active_calibration.gate0 \
  --raw-root results/active_calibration/raw_novel_64 \
  --output-dir results/active_calibration/gate0_80_seed42 \
  --method snapkv --sink 4 --window 32 \
  --compression 0.80 --calibration-size 20 \
  --subset-sizes 4,8,16 --repeats 20 --seed 42
```

输出目录必须尚不存在，避免覆盖旧结果。最重要的文件是 `gate0_results.json`；同目录还会保存 20 问题、均匀预算及每个子集规模第一次抽样的 `.npy` 预算画像。`results/` 已被 Git 忽略，模型与原始曲线不会上传。

## 如何看结果

先看 `full_calibration.mean_proxy_loss` 是否低于 `uniform_budget.mean_proxy_loss`：若连完整 LU 画像也没有优势，当前数据/预算设置不足以展示头分配价值。再看 8 问题随机子集的 `median_delta_proxy_loss` 与 `p90_delta_proxy_loss`，以及 `median_abs_prune_ratio_difference`。如果这些量都接近 0，就说明少量随机问题已足够稳定，主动选题很难有值得研究的收益；若 p90 与中位数均明显大于 0，才有必要进入 Gate 0B：用**不看未选问题 oracle 曲线**的主动选题方法，与相同标定次数的随机/文本多样性方法比较。

注意：这里的 proxy loss 是“被删的中间位置 LU oracle utility 占比”，**不是**答案准确率、Answer-NLL 或真实延迟。`mean_kept_entries` 可核对各画像的总缓存是否相近；预算明显不等时不能把损失差归因于分配。小说样本的 30 问题共用一个 context，因此结果也不能证明跨文本、跨任务泛化。Gate 0B 至少需要独立文本和真实生成质量复核。

## 当前状态与反馈格式

本地合成数据单测已通过；48GB 服务器上的 full-attention 曲线收集与真实 30 问题分析**尚未运行**。服务器运行后请发回终端最后的摘要，以及 `results/active_calibration/gate0_80_seed42/gate0_results.json`。若报错，请附完整 traceback、模型路径所对应的模型家族（不必分享权重）和 `nvidia-smi` 的显存情况。我们再决定是继续 Gate 0B、调整问题池，还是终止此方向。

