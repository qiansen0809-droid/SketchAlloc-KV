# Gate 0A 遗产导入审计（2026-09-28）

## 1. 本次收到的备份

源文件：

```text
C:\Users\Lenovo\Downloads\causalduel_gate0_backup_20260928.tar.gz
```

压缩包 SHA256：

```text
CDD18D1C6DF29DFC167C4E07326B890E458F2BD3BD3E3856AD54A48F1E5752F4
```

导入位置：

```text
results/legacy_causalduel/import_20260928/causalduel_gate0_backup/
```

压缩包路径均为相对路径，没有绝对路径或 `..` 路径穿越项。原始文件未被修改。

## 2. 文件清单与哈希

| 文件 | SHA256 | 用途 |
|---|---|---|
| `gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz` | `5D0F49B67E487FC94041AAF33DB023C5D7F5C1B80C3749046D84846FFDE73C30` | LU 静态预算曲线与边界边际效用 |
| `squality_train_6x5_manifest.json` | `0F28423502200BB574E0B59868CB1C8B0ECD0471F9572B481386E509022BD935` | 6 个 calibration context 的来源、长度与问题清单 |
| `squality_train_6x5.jsonl` | `14D22A0900EA89206F8251A4EBB65A737D218E6798BF740705D2F589BE1CEEB8` | 6 context × 5 questions 的 LU profile 输入 |

## 3. 这批数据包含什么、不包含什么

包含：

- 30 个 SQuALITY calibration pair；
- Llama-3.1-8B 对应的 32 层 × 8 KV head LU profile；
- 80% pruning，即 20% KV retention；
- SnapKV scorer，sink=4，window=32；
- 16-token budget swap 的 remove cost 与 next gain。

不包含：

- `results/gate0/answer_onset/discovery24/*.json`；
- `discovery24_summary.json`；
- 每个 prompt × action 的 `answer_nll_gain_vs_lu`；
- MiniGate 24 条输入和正式结果。

因此，本备份可以验证动作生成过程，但不能计算 best-fixed、family-static、oracle 差值或 cross-over。

## 4. 固定动作字典审计

审计脚本：

```text
evaluation/sketchalloc/audit_legacy_profile.py
```

审计长度：

```text
4096, 5831, 5857, 6350, 6491, 6492, 6595, 8192, 16384, 32768
```

所有长度均得到完全相同、顺序相同的 6 个动作：

| 顺序 | 动作 | 原类别 | LU profile 预测边际增益 |
|---:|---|---|---:|
| 1 | `L0H6 -> L25H0 @16` | lu_promising | `+4.194533709997e-07` |
| 2 | `L27H7 -> L25H0 @16` | lu_promising | `+4.060762311002e-07` |
| 3 | `L22H2 -> L5H2 @16` | near_boundary | `+2.693193816122e-12` |
| 4 | `L17H6 -> L16H6 @16` | near_boundary | `-6.414424546968e-13` |
| 5 | `L18H5 -> L24H0 @16` | near_boundary | `+3.550167567324e-12` |
| 6 | `L27H3 -> L28H1 @16` | low_priority_control | `-2.332124019677e-07` |

完整机器可读结果：

```text
results/sketchalloc/gate0a/legacy_profile_action_audit.json
```

## 5. 当前能得出的结论

### 已完成：动作生成机制的有限重建

在 4K–32K 的测试长度上，当前代码和导入的 LU profile 生成了同一组 6 个动作。旧交接文档也记录过“固定 6 个 candidate”。这只说明旧资产可用于重建候选动作；实际运行过的 24 个 prompt 是否构成完整的 24×6 矩阵，仍须逐个核对原始结果。

本项不计为 Gate 0A 通过，也不能证明 SketchAlloc-KV 的输入自适应收益。

### 尚未通过：prompt 自适应必要性

已有报告中的“79.17% prompt 存在 beneficial swap”和“oracle mean gain = 0.014144”只能证明候选集合中常有正收益动作。它没有回答：

- 是否总是同一个动作产生收益；
- 一个严格留一选择的 best-fixed 能恢复多少 oracle headroom；
- family-static 是否已经足够；
- 同一对动作是否真的在不同 prompt 上发生优劣反转。

因此不能据此宣称 SketchAlloc 所需的 prompt-specific action utility 已成立。

## 6. 下一步所需的最小补充包

服务器只需再打包：

```bash
cd /root/autodl-tmp/CausalDuel-KV
tar -czf /root/autodl-tmp/causalduel_discovery24_results.tar.gz \
  results/gate0/answer_onset/discovery24 \
  results/gate0/answer_onset/discovery24_summary.json
```

无需重新打包模型、数据集、`lu_profile_raw` 或 Hugging Face cache。取回这一个小包后即可直接执行：

1. 逐 JSON 重建稳定 action ID；
2. 确认 24×6 无缺失矩阵；
3. 计算 LOPO best-fixed、family-static、LOFO、cluster bootstrap；
4. 决定 Gate 0A 是 Go、No-Go，还是需要新增独立样本确认。
