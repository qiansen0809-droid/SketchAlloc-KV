# Gate 0A 执行手册：先证明“需要按 prompt 改预算”

> 状态：代码已就绪，等待迁入旧服务器原始结果。
>
> 本阶段不训练 SketchAlloc，不运行低秩恢复，也不声称 Idea 成立。

## 1. Gate 0A 到底在查什么

旧项目在每个 prompt 上分别试了若干预算交换，并报告“经常能找到比 LU-KV 更好的交换”。这还不能证明需要 prompt 自适应，因为可能存在一个固定交换，在所有 prompt 上平均都很好。

Gate 0A 因此只回答两个问题：

1. 旧 24 个 prompt 是否真的执行了同一组动作？
2. 若动作相同，逐 prompt 的 oracle 是否稳定超过严格留一选择的 best-fixed 与 family-static？

第一问不成立时，禁止继续做第二问。

## 2. 需要从旧服务器迁入的最小数据

最小必需目录：

```text
/root/autodl-tmp/CausalDuel-KV/results/gate0/answer_onset/discovery24/
```

建议同时保留：

```text
/root/autodl-tmp/CausalDuel-KV/results/gate0/answer_onset/discovery24_summary.json
/root/autodl-tmp/CausalDuel-KV/results/gate0/minigate/minigate_24.jsonl
/root/autodl-tmp/CausalDuel-KV/results/gate0/minigate/formal24/
/root/autodl-tmp/CausalDuel-KV/results/gate0/lu_profile/
/root/autodl-tmp/CausalDuel-KV/results/gate0/lu_profile_raw/
/root/autodl-tmp/CausalDuel-KV/results/gate0/profile_data/
```

目标位置统一为：

```text
results/legacy_causalduel/gate0/
```

旧目录保持只读，不覆盖、不改名；后续所有 CSV 和统计结果写入 `results/sketchalloc/gate0a/`。

## 3. 第一步：标准化并审计动作字典

在仓库根目录运行：

```bash
python -m evaluation.sketchalloc.normalize_legacy_results \
  --raw-dir results/legacy_causalduel/gate0/answer_onset/discovery24 \
  --output-csv results/sketchalloc/gate0a/legacy24_actions.csv \
  --audit-json results/sketchalloc/gate0a/legacy24_coverage.json
```

脚本不会相信旧候选名称，而是根据以下信息重建动作：

```text
action_id = donor layer/head -> receiver layer/head @ amount
```

重点查看 `legacy24_coverage.json`：

- `complete_action_matrix=true`：每个 prompt 都有完全相同的一组动作，可以进入下一步；
- `complete_action_matrix=false`：旧结果只能证明“各 prompt 的局部候选里存在好动作”，不能比较 best-fixed；需补跑固定动作集；
- `action_counts`：每个动作覆盖了多少 prompt，用于定位动作漂移。

## 4. 第二步：严格 Gate 0A 分析

仅当动作矩阵完整时运行：

```bash
python -m evaluation.sketchalloc.analyze_gate0a \
  --input results/sketchalloc/gate0a/legacy24_actions.csv \
  --output-json results/sketchalloc/gate0a/legacy24_gate0a.json \
  --bootstrap 5000
```

分析器强制执行：

- LU-KV 永远是可回退的零收益动作；
- best-fixed 用 leave-one-prompt-out 选择，不能用当前样本自己的标签；
- family-static 只能使用同任务族的其他 prompt；
- leave-one-family-out 检查固定动作能否跨任务族泛化；
- bootstrap 以 `context_cluster` 为抽样单位；
- 动作排序 cross-over 按同一对动作在不同 prompt 上的优劣反转计算。

## 5. 如何读结果

### 可以继续固定动作复现实验

至少应看到：

- `oracle_minus_best_fixed_ci95` 下界大于 0；
- `pairwise_action_crossover_rate >= 0.30`；
- family-static 不能恢复大多数 oracle headroom；
- leave-one-family-out 后结论方向不翻转。

这仍只是 24 样本 screening。下一步不是立即写论文，而是冻结 6–8 个共享动作，在新样本上复现。

### 暂停 prompt 自适应 Idea

出现任一情况就暂停：

- 动作矩阵不完整，且固定动作补跑后 oracle 优势消失；
- best-fixed 或 family-static 已恢复绝大多数 oracle headroom；
- 排序 cross-over 很少，说明只是同一个动作普遍好；
- 优势由单一任务族或少数重复 context 驱动。

此时更合理的论文方向是重新校准 LU-KV 的静态 profile，而不是继续做 SketchAlloc。

## 6. 当前执行状态

- [x] 旧 JSON → 稳定动作 ID 的转换器；
- [x] 动作矩形覆盖审计；
- [x] LU 回退、LOPO best-fixed、LOPO family-static、LOFO 与 cluster bootstrap；
- [x] 合成单元测试；
- [x] 导入 LU marginal profile 与 30 条 SQuALITY calibration 输入；
- [x] 在 4K–32K 共 10 个测试长度上重建候选，观察到生成器给出同一组 6 个动作；实际实验动作尚待原始 JSON 核对；
- [ ] 迁入 AutoDL 原始结果；
- [ ] 生成真实 `legacy24_coverage.json`；
- [ ] 根据覆盖审计决定直接分析或补跑固定动作。

当前缺的不是算法代码或 LU profile，而是旧服务器上的原始 `discovery24/*.json`。详见
`docs/gate0a_import_audit_2026-09-28.md`。
