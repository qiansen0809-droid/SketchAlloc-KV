# SketchAlloc-KV Gate 0：结构、可识别性与真实收益三重生死门

> 文档性质：预注册草案，不是 ResearchPilot 阶段 C 的完整 Part 3。
>
> 日期：2026-09-28
>
> 对应 Idea：`docs/idea_report.md`
>
> 目标：先用 CausalDuel-KV 遗产回答“是否值得做”，再决定是否投入新 scorer、物理 cache 和大规模实验。

## 0. 总判定逻辑

SketchAlloc-KV 必须连续通过三道门：

```text
Gate 0A：真的需要 prompt 自适应吗？
    ↓ 通过
Gate 0B：动作效用残差真的低秩吗？
    ↓ 通过
Gate 0C：少量无答案机械探针能识别该低秩坐标吗？
    ↓ 通过
才进入正式多模型、长上下文和物理系统实验
```

任何一门失败都停止当前 idea。禁止在同一 discovery set 上反复换动作、换秩、换指标直到通过。

## 1. 先冻结旧项目结论

旧 CausalDuel 结果只允许支持：

- query-tail 和 answer-onset 行为相似度不能稳定判断 swap utility；
- 当前候选集合中经常存在优于 LU 的动作；
- 现有预算交换和 Answer-NLL 评估链路可运行。

它不允许支持：

- 79.17% 的 prompt 必须使用不同动作；
- 任何 prompt-only proxy 已经有效；
- per-prompt oracle 的收益无法由一个固定 swap 解释；
- 现有 24 个样本足以证明泛化。

## 2. Gate 0A：排除“最佳固定动作”解释

### 2.1 输入数据

优先迁移服务器上的 24×6 共 144 个候选结果，并为每条记录补充稳定动作 ID：

```text
action_id = donor_layer:donor_kv_head -> receiver_layer:receiver_kv_head @ swap_size
```

若六个候选不是跨 prompt 相同动作，则这批数据只能算先导，不能直接形成矩阵。需要根据 LU 静态边界重新选择 6–8 个所有 prompt 共享的动作，重跑 24 个样本。

### 2.2 必须比较的五个水平

1. LU-KV 基线；
2. 随机合法动作；
3. 全 discovery prompts 上平均收益最高的 best-fixed 动作；
4. 每个任务族一个 family-static 动作；
5. 每个 prompt 的 local oracle 动作。

### 2.3 统计量

- `oracle - best_fixed` 的 prompt-cluster bootstrap 95% CI；
- `oracle - family_static`；
- prompt 间最佳动作的分布熵；
- cross-over rate：至少两个动作在不同 prompt 上互相反转优劣的样本比例；
- leave-one-task-family-out 时 best-fixed 的泛化；
- 选择动作的 Answer-NLL gain 与任务 score gain。

### 2.4 通过条件

进入 Gate 0B 前必须同时满足：

- `oracle - best_fixed` 的均值为正且 prompt-cluster bootstrap 95% CI 下界大于 0；
- `oracle - family_static` 同样为正，或 family-static 无法解释至少 70% 的 oracle-over-LU headroom；
- 至少 30% prompt 出现可重复的动作排序 cross-over，而不是同一动作普遍占优；
- 结论在去掉任一任务族后方向不翻转。

若不满足，结论是“旧 LU profile 或动作字典需要静态校准”，而不是 prompt-adaptive 新算法。

## 3. Gate 0B：低秩结构检验

### 3.1 扩展数据

24×6 只用于摸底。正式结构门使用：

- 96 个独立 prompt；
- 24 个跨 prompt 固定动作；
- 任务组成：32 RULER retrieval、32 LongBench 单文档 QA、32 多文档 QA；
- 主长度约 8K；
- 主压缩率 20% retention；
- 所有候选严格等总预算、同 scorer、同最小保护窗口。

数据隔离：

- discovery 48；
- calibration 24；
- held-out 24；
- 同一原始文档、RULER 模板族和派生问题不得跨集合。

若预算允许，held-out 扩到 48。任何正式阈值在看 held-out 标签前冻结。

### 3.2 效用矩阵

每个单元为相对 LU 的 gain：

```text
U[prompt, action] = NLL_LU - NLL_action
```

另存任务原始分数差，不用它拟合主模型。

建立三种矩阵：

1. 原始 `U`；
2. 去动作均值后的 `U - mean_action`；
3. 去任务族静态均值后的残差。

只有第三种仍有稳定结构，才说明不是把任务标签换个形式编码进去。

### 3.3 结构检验

- discovery 上拟合 rank 1–8 截断 SVD/概率 PCA；
- calibration 选择最小可用秩；
- held-out 计算重建 `R²`、RMSE、within-prompt Spearman、top-action regret；
- 对行/列置换做 parallel analysis；
- 对 prompt 聚类 bootstrap 奇异值与右奇异向量稳定性；
- 做 leave-one-family-out 和跨长度外推诊断。

### 3.4 通过条件

- rank ≤ 4 在 held-out 上解释至少 70% 的 family-centered utility variance；
- held-out within-prompt Spearman 中位数至少 0.60；
- 使用完整 oracle 行坐标重建时，top-action regret 不超过 oracle headroom 的 30%；
- 右奇异向量在 bootstrap 中具有可对齐的稳定结构，而不是由少数异常 prompt 决定；
- family-centered 结果仍明显优于动作均值和随机低秩基。

若只能在原始矩阵上低秩、去掉任务族均值后消失，则得到的是任务分类，不是新的 prompt-level 分配结构。

## 4. Gate 0C：少量探针的可识别性

### 4.1 探针候选

对每个固定动作，在不使用 gold answer 的情况下提取：

- donor/receiver query-token attention-output delta；
- 1/2/4 层后的 normalized residual delta；
- donor loss 与 receiver gain 的局部 error asymmetry；
- KVSculpt-style local MSE delta；
- 运行时间与临时显存。

旧 CausalDuel 的 probability overlap、KL、JS、top-k 和 pseudo-answer NLL 只作为失败 baseline，不作为主候选。

### 4.2 探针集合选择

仅在 discovery 上比较：

- 随机 m 个动作；
- LU marginal 最大/最接近边界的 m 个动作；
- leverage score；
- pivoted QR；
- D-optimal design。

`m ∈ {2, 4, 6, 8}`。在 calibration 上一次性冻结：探针特征、秩 `r`、探针数 `m`、岭回归系数与回退阈值。

### 4.3 强制对照

必须包含：

- LU static；
- best-fixed；
- family-static；
- DynamicKV/EntroKV 可计算分数；
- all-action KVSculpt-style local proxy；
- 与 SketchAlloc 参数量相近的 MetaKV-style prompt feature predictor；
- random probes；
- oracle latent coordinates；
- local oracle action。

没有 prompt-feature predictor 对照，无法证明收益来自“主动干预”而非普通监督学习。

### 4.4 主评价指标

- held-out top-action regret；
- recovery of oracle-over-best-fixed headroom；
- selected action 的正收益率、伤害率和平均 Answer-NLL gain；
- within-prompt Spearman；
- 相对 all-action pilot 的探针数、FLOPs、wall-clock 与显存；
- 任务原始指标的 paired gain；
- fallback 覆盖率与 fallback 后的净收益。

### 4.5 通过条件

必须同时满足：

- `m ≤ 6` 时恢复至少 60% 的 oracle-over-best-fixed Answer-NLL headroom；
- 相对 best-fixed 的 paired mean gain 95% CI 下界大于 0；
- top-action regret 比同参数量 prompt-feature predictor 至少降低 15%；
- 正收益 prompt 比例至少 60%，伤害比例不高于 25%；
- 至少两个任务族的任务原始指标不劣于 best-fixed，合并指标有正向 paired effect；
- 探针成本不超过 all-action pilot 的 30%，并且根据实测算子外推，8K TTFT 增量不高于 10%。

这些阈值是进入正式实验的门槛，不是最终论文声称。若结果接近门槛但置信区间跨零，应扩充未查看标签的新样本，而不是降低标准。

## 5. 物理可执行性 Gate -1

Gate 0C 通过后、正式多模型实验前，再做物理后端审计：

1. 预算变化是否真实减少已分配 KV entry/page，而非只写 mask；
2. GQA 下动作单元是否是 KV head/group，而非虚构的 query head 独立缓存；
3. donor 减少与 receiver 增加是否严格等物理字节；
4. 探针能否共享 FullKV prefix，是否导致临时显存随 `m` 线性增长；
5. 最终选择后能否直接继续 decode，还是必须重新 prefill。

无法落到物理 cache 时，算法结果只能写成分析/模拟，不能声称真实显存和吞吐收益。

## 6. 旧资产迁移清单

### P0 必须从服务器带入新项目

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

复制后生成 SHA256 manifest，原目录保持只读。旧结果只用于 Gate 0A discovery，不进入新 held-out。

### 可直接继承的代码

```text
evaluation/gate0/build_candidates.py
evaluation/gate0/minigate_runtime.py
evaluation/gate0/prepare_minigate.py
evaluation/gate0/prepare_squality_profile.py
evaluation/gate0/metrics.py
evaluation/curve_data/
kvpress/presses/LU_press.py
kvpress/attention_patch.py
tests/test_gate0_candidates.py
tests/test_gate0_marginals.py
tests/test_gate0_metrics.py
```

### 只归档、不再作为主链路

```text
suffix_replay.py
run_answer_onset.py
summarize_answer_onset.py
behavior-overlap selector
successive behavioral racing
```

## 7. 推荐执行顺序与算力止损

1. 不开 GPU：迁移 144 条旧结果，统一 action ID，跑 Gate 0A。
2. 若动作不统一：只补跑固定 6–8 动作 × 24 prompt。
3. Gate 0A 通过后：先跑 48 prompts × 12 actions 的半规模谱诊断。
4. 只有出现稳定低秩趋势，扩为 96 × 24。
5. Gate 0B 通过后才实现局部 residual probe。
6. 先比较 m=4/6 与随机、prompt-feature predictor；通过才做物理后端。
7. 物理 Gate -1 通过后才上 16K/32K、第二模型和正式 baseline。

## 8. 预注册输出文件

每个样本至少保存：

```text
prompt_id, document_id, task_family, token_length
model, scorer, retention, swap_size, seed
action_id, donor, receiver, total_keep_count
lu_answer_nll, action_answer_nll, answer_nll_gain
lu_task_score, action_task_score, task_score_gain
local_probe_features
wall_clock_ms, peak_allocated_bytes
split, code_commit, data_manifest_hash
```

矩阵分析必须记录缺失动作；不得用均值静默填补失败 run。缺失率超过 5% 或与动作/任务显著相关时，先修工程，不解释谱结构。

## 9. Gate 0 最终判定表

| 门 | Go | No-Go 后的结论 |
|---|---|---|
| 0A 自适应必要性 | oracle 显著超过 best-fixed/family-static，并有 cross-over | 做静态 LU profile 校准，停止在线 idea |
| 0B 低维结构 | held-out family-centered 残差可由 rank≤4 稳定重建 | 停止低秩恢复，不上大预测器 |
| 0C 可识别性 | ≤6 探针优于 predictor/random 并恢复多数 headroom | 结构有分析价值，但算法不可部署 |
| -1 物理落地 | 实际 page/byte 守恒且开销可接受 | 不写系统收益，不进入正式论文主线 |

只有四门全过，才值得投入 CCF-C 论文级完整实验。
