# SketchAlloc-KV 新颖性碰撞检查

> 检索日期：2026-09-28
>
> 结论级别：**截至检索日未发现完整同构工作，但新颖性是有条件的，必须通过 Gate 0 才能成立。**
>
> 暂定方法名：SketchAlloc-KV。名称只用于项目管理，论文标题在 Gate 0 通过前不冻结。

## 1. 待检查的最小创新声明

本项目不把“prompt-specific 分配”“低秩”“边际效用”“局部重构误差”中的任何一个单独声明为创新。待检查的是下面这一完整组合：

1. 将同一压缩后端下、严格等总预算的 donor-receiver 动作组成固定动作字典；
2. 把不同 prompt 对这些动作的真实边际收益写成 `prompt × action` 效用矩阵；
3. 检验静态平均效用之外的样本级残差是否具有低有效秩；
4. 用 QR/D-optimal sensor placement 挑选少量代表性“哨兵动作”；
5. 在线只对这些动作做局部机械干预测量，据此识别当前 prompt 的潜在效用坐标；
6. 重建全部动作的边际收益，再由固定总预算求解器分配缓存；
7. 当低秩重建残差或外推距离过大时回退 LU-KV，但不把风险认证本身写成贡献。

直白地说：我们不是给所有 head 重新打一次分，而是先找出“测哪几个动作最能判断当前 prompt 属于哪种预算需求”，再利用结构恢复剩余动作。

## 2. 直接近邻逐项碰撞

| 工作 | 已经覆盖的内容 | 与 SketchAlloc-KV 的实质区别 | 判定 |
|---|---|---|---|
| [LU-KV](https://arxiv.org/abs/2602.08585) | 离线测量长程边际效用，凸包/PAVA 后形成静态 head 级画像 | 本项目继承其动作定义、先验和求解器，只研究静态画像之外的样本级残差是否可由少量干预恢复 | 直接母体，必须超过 |
| [KVSculpt](https://arxiv.org/abs/2603.27819) | 对每个 layer/head 做 pilot compression，用局部 MSE 分配预算 | KVSculpt 直接测全部组件并按 MSE 映射预算；本项目只测少数经实验设计选出的动作，目标是恢复端到端边际效用面 | 最近碰撞，需强对照 |
| [MetaKV](https://arxiv.org/abs/2609.07966) | 用 prompt 特征预测每套完整压缩配置的正确率、延迟和显存，再选配置 | MetaKV 在方法/配置级选择；本项目固定后端、scorer 与总预算，在 layer/head 动作空间内用机械干预识别效用坐标 | 不同粒度，但要加 MetaKV-style predictor 对照 |
| [Tangram](https://arxiv.org/abs/2606.06302) | 发现 head 排序和预算比例在多轮场景中较稳定，并使用静态画像消除运行时开销 | 该结论直接挑战 prompt 自适应的必要性；本项目只有在“样本级 oracle 明显超过最佳静态动作”时才成立 | 反方关键证据 |
| DynamicKV / EntroKV | 从当前输入的 attention 或 entropy 直接产生 layer/head 预算 | 它们逐单元计算代理并直接映射预算，不建模动作效用面的低维结构，也不做主动传感器选择 | 部分重叠，非同构 |
| CoKV | 用 Shapley 型联合贡献衡量 head interaction | CoKV 的贡献值按任务/校准集估计；本项目研究 prompt 级固定预算动作的潜在结构与稀疏识别 | 不同，但不能声称首次考虑交互 |
| GraceKV | 把 coverage/resolution 动作放入统一边际收益队列 | GraceKV 已经把分配写成动作竞争；本项目的新增点只能是“少量干预重建样本级动作效用”，不能是全局资源流本身 | 动作表示已有先例 |
| LAVa / CriticalKV / ReST-KV / RippleKV | 用 attention-output 损失、形式化扰动或注入扰动衡量单元敏感性 | 这些方法覆盖“扰动/输出误差可作为信号”；本项目可把它们作为哨兵观测量，但不能据此声称新指标 | 探针来源已碰撞 |
| [Linear Predictability of Attention Heads](https://openreview.net/pdf?id=rsQm4VxNKV) | 从少数 peer head 线性重建其他 head 的 K/V 激活，并据此压缩 | 它重建的是激活状态，不是 prompt 对预算动作的边际效用曲线 | 关键词近似，目标不同 |
| [Risk-Controlled KV-Cache Eviction](https://arxiv.org/abs/2609.27981) | 用有限样本风险认证选择保留策略，无策略通过时回退 FullKV | 风险目标、有限样本保证和保守回退已被占据；本项目不得把 conformal/risk fallback 作为主贡献 | 已撞线，删除该贡献 |
| [Marginal utility, matrix factorization, and the KV cache](https://arxiv.org/abs/2609.20068) | 理论上联系边际效用、表示的谱和低秩压缩 | 该文分解表示/协方差，不做 prompt-action 效用矩阵的稀疏传感与分配恢复 | 标题碰撞，机制不同 |

## 3. 目前没有检索到的完整组合

使用下列关键词及其组合检索 arXiv、OpenReview 和官方会议页面：

- `KV cache low-rank utility profile allocation matrix completion`
- `KV cache sentinel heads active probing allocation`
- `KV cache sparse probes marginal utility`
- `KV cache prompt-conditioned utility subspace`
- `KV cache profile reconstruction leverage score`

截至 2026-09-28，未找到同时满足以下条件的论文：

- 对象是同一后端内部的 budget-preserving layer/head 动作，而非 token 评分、整套配置或 KV 激活；
- 学习的是 prompt-action 边际效用面的低维基；
- 在线通过主动挑选的少量干预识别潜在坐标；
- 重建完整效用面后再做固定总预算优化。

这只能支持“当前未发现直接重复”，不能证明全球范围绝对无人做过。投稿前仍需按标题、摘要、引用链和同期 OpenReview 再检索一次。

## 4. 不能写进论文的伪创新

以下表述已经被相关工作覆盖，禁止作为贡献：

- “首次发现不同 layer/head 的预算重要性不同”；
- “首次使用边际效用分配 KV cache”；
- “首次使用 prompt-specific 动态预算”；
- “首次使用局部重构误差/输出扰动”；
- “首次使用低秩结构压缩 KV cache”；
- “首次根据不确定性回退静态方案”；
- “首次把配置选择视为预测问题”。

## 5. 新颖性成立的必要证据

SketchAlloc-KV 只有同时满足下面四项，才够形成独立论文：

1. **存在适应空间**：per-prompt oracle 必须明显优于 LU-KV、最佳固定动作和任务族静态动作；只优于 LU 不够。
2. **存在可恢复结构**：样本级残差矩阵在 held-out prompt 上确实表现出低有效秩，不能只在训练矩阵上画漂亮奇异值。
3. **少量探针有价值**：4–8 个探针应显著优于同参数量的 prompt-feature predictor、随机探针和只用 LU prior，并恢复大部分 oracle headroom。
4. **真实质量与成本成立**：重建分配不仅改善 Answer-NLL，还应改善/不伤害任务分数，并在真实物理缓存后端上保持可接受的 TTFT、显存和吞吐。

任何一项失败，都不应通过改名字继续包装同一故事。

## 6. 当前判定

**判定：可以进入 Gate 0，但不能直接进入正式论文实验。**

它比 CausalDuel-KV 更可行的原因不是“理论更高级”，而是把最难的问题拆成了三个可单独否证的子问题：有没有真正的 prompt-specific headroom、效用面是否低秩、少量可观测干预能否识别该低秩坐标。旧项目已经提供动作构造与真实收益标签，因此第一轮检验不必重新搭环境。

它的最大风险同样清楚：Tangram 的输入不变排序可能意味着样本级残差太小；即使低秩，局部机械观测也可能无法识别对任务真正有用的坐标。Gate 0 的作用就是在继续投入前把这两种失败暴露出来。
