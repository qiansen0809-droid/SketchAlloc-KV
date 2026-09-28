# SketchAlloc-KV Idea Report

> 生成时间：2026-09-28
>
> 状态：`PENDING_PART2_REVIEW`
>
> 研究阶段：ResearchPilot B（Idea 深化）；当前只给出 Part 1 与 Part 2。用户确认后才进入正式阶段 C。
>
> 一句话概括：**LU-KV 给出平均情况下的预算先验；SketchAlloc-KV 不再尝试从 prompt 文本一次性猜完整预算，而是主动测量少数最有辨识力的预算动作，识别当前 prompt 的潜在效用类型，再重建全部动作的边际收益。**

## Part 1 Topic Overview

### 1. Motivation

KV-cache 预算分配已经从 PyramidKV 的固定层间金字塔，发展到 DynamicKV、Ada-KV、EntroKV 的输入相关 layer/head 分配，再发展到 LU-KV 的长程边际效用曲线。问题已经不是“预算需不需要非均匀”，而是“有限测量成本下，怎样知道当前 prompt 真正需要哪一种非均匀分配”。

LU-KV 是一个很强的起点：它离线测量每个 head 在不同预算点的长程效用，经凸包/PAVA 得到稳定的静态分配。它避免了 attention mass 或 entropy 在不同 head 之间不可比的问题，但最终仍把多条样本级曲线平均为一个画像。这意味着它回答的是“通常怎么分”，不保证回答“这个 prompt 怎么分”。

CausalDuel-KV 曾试图用 FullKV/CompressedKV 的 query-tail 或 answer-onset 行为相似度直接选择局部 swap。24 个 prompt、144 个候选的结果显示，该信号接近随机，选择后平均收益为负，因而这条路线已经终止。不过遗留结果还暴露了一个值得更严谨检查的现象：很多 prompt 的候选集中存在优于 LU 的动作。该现象不能直接叫作 prompt-specific headroom，因为也可能只是某个固定 swap 普遍更好，或者校准分布与测试分布不匹配。新项目首先要把这三种解释拆开。

另一方面，KVSculpt 会对全部 layer/head 做一次 pilot compression，MetaKV 会根据 prompt 特征预测整套配置，Tangram 则给出相反证据：许多 head 的保留排序近似输入不变。直接再训练一个“prompt → 每个 head 预算”的预测器，既容易与现有工作重叠，也需要大量标签。SketchAlloc-KV 因而借鉴系统辨识和传感器选择：若不同 prompt 的动作效用虽然会变化，但变化主要落在少数潜在方向上，那么没有必要测量全部动作；只需挑选少数最能区分这些方向的哨兵干预，就可能恢复整个效用面。

#### 直白解释

假设有 32 种“从 A 拿一页给 B”的合法动作。传统动态方法会给 32 个动作各算一个分数；普通预测器会看 prompt 后直接猜 32 个收益；CausalDuel 会逐个试动作并比较输出。SketchAlloc-KV 想做的是：先离线发现这 32 个收益实际上主要由 3 种隐藏因素控制，再挑出最能区分这 3 种因素的 4–6 个动作。新 prompt 来时只试这几个代表动作，判断它更像哪种需求，再推断其他动作的收益。

### 2. Research Questions

#### RQ1：真正存在 prompt-specific 的可利用分配空间吗？

在相同 token scorer、压缩后端和总预算下，per-prompt oracle 是否不仅优于 LU-KV，还显著优于“全体 prompt 最佳固定动作”和“任务族最佳静态动作”？

这是首要问题。若最佳固定 swap 已能解释旧结果，便不存在在线自适应的必要性。

#### RQ2：样本级边际效用面是否具有可泛化的低维结构？

把 prompt 对固定动作字典的 Answer-NLL/任务收益组成矩阵后，去除 LU prior 与动作均值，held-out prompt 的残差能否由很少的潜在因子重建？

训练矩阵上的高 explained variance 不算通过，必须在按 prompt、文档和任务族隔离的测试上成立。

#### RQ3：少量机械干预能否识别潜在坐标？

只测 4–8 个经 QR 或 D-optimal design 选出的哨兵动作，并读取它们的局部 attention-output / residual 响应，能否比 prompt-feature predictor、随机探针和全局静态画像更准确地恢复当前 prompt 的动作排序？

#### RQ4：恢复出的分配能否带来真实质量收益且成本合理？

在物理 KV cache 后端中，SketchAlloc-KV 是否能改善 gold-answer NLL 与任务指标，同时把额外 TTFT、峰值显存和探针开销控制在可部署范围？

### 3. 核心工作与新颖性边界

| 工作 | 与本项目的关系 | 本项目不能重复声称的内容 |
|---|---|---|
| LU-KV | 静态效用先验、动作边界、全局求解器 | 边际效用与凸优化 |
| KVSculpt | 全组件 pilot MSE 是最强直接近邻 | 用 pilot error 做预算 |
| MetaKV | prompt 级配置预测对照 | 输入自适应配置选择 |
| Tangram | 静态 head 排序可能已足够，是反方基线 | 预算异质性与系统落地 |
| DynamicKV / EntroKV | 便宜输入代理基线 | attention/entropy 驱动动态分配 |
| CoKV / GraceKV | 联合贡献和动作竞争近邻 | 首次考虑交互或全局动作队列 |
| LAVa / CriticalKV / ReST-KV / RippleKV | 局部扰动观测候选 | 输出扰动本身的新颖性 |
| Risk-Controlled KV | 已覆盖风险目标与认证回退 | 风险控制/保守回退作为主贡献 |

完整碰撞检查见 `docs/collision_check_sketchalloc_2026-09-28.md`。当前没有找到“固定预算动作效用面 + 主动哨兵干预 + 低秩重建 + 全局预算求解”的同构论文；该结论截至 2026-09-28 有效，投稿前必须更新。

## Part 2 Idea Design

### 1. 核心假设

设 LU-KV 给出的基线预算为 `b0`，固定动作字典包含 `A` 个合法动作。每个动作可以是一次预算守恒的 donor-receiver page swap，也可以是同一求解路径上的一个边际增减动作。对 prompt `x`，动作 `a` 的真实收益记为：

```text
U(x, a) = Loss(x, b0) - Loss(x, apply(b0, a))
```

收益为正表示该动作比 LU 基线好。在线阶段看不到 gold answer，因此 `U` 只用于离线建立和检验结构。

先去除每个动作的静态平均收益：

```text
R(x, a) = U(x, a) - mean_train[U(·, a)]
```

核心可证伪假设是：`R` 不是任意高维噪声，而近似满足：

```text
R ≈ Z Vᵀ
```

其中 `V` 是少量跨 prompt 共享的效用方向，`Z(x)` 是当前 prompt 在这些方向上的坐标。若有效秩为 3–4，就可能用 4–8 个有辨识力的观测恢复 `Z(x)`，再重建全部动作收益。

### 2. 方法总览

```text
离线动作字典与真实效用标签
          ↓
去除 LU / 最佳静态动作能解释的部分
          ↓
学习 prompt-action 效用残差基 V
          ↓
用 QR / D-optimal design 选择少量哨兵动作
          ↓
新 prompt 上只执行哨兵局部干预，收集机械响应 z
          ↓
求解潜在坐标 Z(x) 并重建全部动作收益
          ↓
固定总预算求解；异常样本回退 LU
```

### 3. 模块一：固定预算动作字典

动作字典必须跨 prompt 固定，不能每个样本临时挑完全不同的 donor/receiver，否则无法形成可比较矩阵。

以 LU-KV 静态 profile 为中心：

- 从 LU remove cost 最低的合法 donor 中选一组；
- 从 LU next gain 最高或接近决策边界的 receiver 中选一组；
- 形成 24–32 个 budget-preserving swaps；
- 覆盖 LU 看好的动作、边界不确定动作、跨层与层内动作；
- 所有动作保持完全相同的总 keep count、scorer、sink/window 下限和压缩率。

Gate 0 只做单步动作。只有单步选择有效，正式方法才考虑连续执行 2–4 步并在每一步重新估计剩余边际收益。

### 4. 模块二：效用图谱与低维基

在 discovery prompts 上离线评估所有动作：

- 主标签：相对 LU 的 gold-answer token 平均 NLL 改善；
- 次标签：任务原始指标变化；
- 诊断标签：局部 attention-output error、residual error、LU marginal delta。

先比较 LU、最佳固定动作与任务族静态动作，再对仍未解释的残差做截断 SVD/概率 PCA。有效秩不由肉眼决定，而用 held-out 重建误差、parallel analysis 和 bootstrap spectrum 稳定性共同确定。

### 5. 模块三：主动选择哨兵动作

若效用基为 `V ∈ R^(A×r)`，从动作行中挑 `m` 个最能辨识 `r` 维坐标的动作。Gate 0 比较：

- 随机动作；
- LU 边界最高动作；
- leverage-score 选择；
- pivoted QR；
- D-optimal design。

选择算法只使用 discovery 数据，held-out prompt 上动作集合完全冻结。

### 6. 模块四：无答案的机械观测

新 prompt 上不能计算真实 `U`。每个哨兵动作只运行轻量局部干预，并记录以下候选观测：

- 被交换单元在 query-bearing token 上的 normalized attention-output delta；
- 向后传播 1/2/4 层后的 residual-stream delta；
- receiver 获得预算后与 donor 失去预算后的 error asymmetry；
- 局部 MSE 相对 LU 的变化。

这些量均不被声明为新指标；它们是识别潜在坐标的传感器读数。离线学习从哨兵读数到 `Z(x)` 的小型线性/岭回归观测模型。与 UtilityCal-KV 的区别在于：

1. 不从 prompt embedding 直接预测每个 head 的绝对收益；
2. 输入是对当前 cache 做真实干预后的响应；
3. 输出被约束在已验证的低维效用子空间；
4. 探针位置由可辨识性而非人工经验选择；
5. 可以用观测残差直接判断当前样本是否超出子空间。

如果非线性模型明显优于线性模型，Gate 0 也只把它作为失败信号：这说明低维可识别故事不足，不能靠堆一个大预测器救活。

### 7. 模块五：重建、分配与回退

根据哨兵读数估计潜在坐标，再得到每个动作的预测收益：

```text
Û(x, ·) = static_prior + V Z_hat(x)
```

单步 Gate 0 选择最高预测正收益动作；若所有动作非正，则保留 LU。正式版本可将重建的边际曲线送回 LU 的凸优化/贪心求解器，在严格总预算下生成完整预算。

回退只作为工程保护：当观测重建残差、leverage 外推距离或多次 probe 的一致性超过 calibration 阈值时使用 LU。由于 Risk-Controlled KV 已经覆盖风险合同与有限样本认证，本项目不声称提供新的风险保证。

### 8. 与 CausalDuel-KV 的关系

#### 可继承

- LU keep-count override 与 attention patch；
- LU marginal profile 导出；
- budget-preserving swap 与候选合法性检查；
- SQuALITY profile、MiniGate、RULER/LongBench 数据准备；
- FullKV/LU/candidate cache 构造；
- gold-answer NLL、任务指标、prompt-cluster bootstrap；
- 预算守恒、sink/window 下限和 GQA 单元测试。

#### 必须停用

- query-tail probability overlap 作为 selector；
- answer-onset/pseudo-answer trace；
- KL/JS/top-k agreement 的在线选候选故事；
- “79.17% prompt 有 beneficial swap，所以存在 prompt adaptivity”的结论；
- cache-fork/successive racing 作为主要方法贡献。

旧行为指标可以保留为负结果和 baseline，但不能参与新方法调参。

### 9. 预期贡献

若 Gate 0 通过，论文的独立贡献应是：

1. 揭示并严格验证 prompt-action KV 预算边际效用残差的低维结构；
2. 提出基于主动传感器选择的稀疏机械干预，用少量观测恢复完整样本级效用面；
3. 将恢复出的边际曲线接入严格预算守恒的分配求解器，在少量额外开销下优于静态效用与全组件 pilot；
4. 给出该结构何时失效、何时应回退静态 profile 的实证边界。

其中任何一项在 Gate 0 前都只能写作“计划验证”，不能写成已经发现。

### 10. 失败模式与转向规则

- 若 best-fixed swap 已接近 per-prompt oracle：停止在线自适应，研究静态 profile 校准，不继续 SketchAlloc。
- 若残差矩阵没有稳定低秩：停止低秩恢复，不用更大网络硬拟合。
- 若 oracle 低秩恢复有效但机械观测不能识别坐标：可保留为分析论文素材，但不够形成当前算法论文。
- 若质量有效但必须测量超过 25% 动作或 TTFT 过高：不进入系统实现，重新评估是否值得做离线 task-profile。
- 若只改善 NLL、不改善任务指标：不能宣传端到端收益，应增加更合适的真实效用标签或停题。

### 11. CCF-C 可发表性判断

仅有“效用矩阵低秩”或一个 SVD 图，不足以投 CCF-C。若能完成“新现象 + 主动探针算法 + 严格 held-out 改善 + 真实物理系统开销”这四段闭环，且在至少两个模型、三个任务族、8K/16K/32K 和两档压缩率上超过 LU-KV、KVSculpt-style pilot、DynamicKV/EntroKV 与 MetaKV-style predictor，则有形成 CCF-C 独立论文的合理可能；若还能证明稀疏探针相对全组件 pilot 的数量级成本优势，具备进一步冲击更高档 venue 的空间。

### 12. 阶段 B 确认卡

请用户确认以下内容后再进入阶段 C：

- [ ] 接受暂名 SketchAlloc-KV；名称可在 Gate 0 后修改。
- [ ] 接受“先证明 best-fixed 不能解释旧 headroom”，而不是直接把 79.17% 当作动态性证据。
- [ ] 接受动作字典固定为 24–32 个预算守恒 swap。
- [ ] 接受主方法为“低维效用面 + 主动哨兵干预”，不再优化 FullKV 行为相似度。
- [ ] 接受 Gate 0 失败即停题，不用更大预测器挽救。
- [ ] 接受 gold answer 只用于离线标签和验证，在线方法不读取答案。

用户确认前，`docs/gate0_sketchalloc_kv.md` 只作为预注册草案，不等同于正式 Part 3。
