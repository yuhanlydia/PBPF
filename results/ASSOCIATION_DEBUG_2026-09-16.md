# A-PBPF 调试与设计验证（2026-09-16）

结论：显式测试 × diagnosis 交互解决了可控任务的学习失败；真实 RunBugRun 开发集的 association 仍未成立。没有重新评估原测试集，也没有运行 repair。

## 可复现的问题与修改

1. 旧 proposal 只读第一条证据，有限粒子近似对配对顺序敏感。新增 prefix importance sampling，每个前缀读取全部已见配对、采用交换不变的集合编码，并保留 prior/proposal 修正。这是新的推断消融，不是原 SMC/FIVO 的同义改写。
2. 旧 difficulty 约束比较五类输出概率，不能约束完整 latent 分布。新增标准化 difficulty 粒子的加权 RBF MMD；明确它不是精确 KL，也不保证可辨识性。
3. 统一 evidence/future 的 nats/test 量纲。association comparator 停梯度，阻断直接通过恶化 shuffled 分支降低 hinge 的梯度路径；共享参数仍可能使 shuffled 变差，必须同时考察 aligned 质量。
4. 在绝对预测质量约束内按开发集 association 与配对不变性筛选 checkpoint；没有合格 checkpoint 时明确失败。另报三个 Monte Carlo 随机数复核，避免把一次采样的有利结果当结论。
5. 新增显式双线性交互 decoder；同时加入具有 outcome-conditioned test summaries 的确定性交互基线，避免把结构优势误判成粒子优势。
6. 修复旧半精度 SMC 在零偏移边界选到 native precision 零质量粒子的问题；以固定噪声替换依赖 Torch 版本的随机种子测试。它不能解释此前单精度 pilot 的失败。

## 可控机制实验

每段可见历史均有 2 PASS / 2 FAIL。随机隐藏方向决定哪些带符号测试失败，代码/任务噪声不透露方向。只靠直方图无法区分；配对证据可以。模型不读取隐藏诊断标签。此处成功仅证明可学习性。

| 版本 | 种子 | 开发 NLL ↓ | shuffle gap ↑ | 开发筛选 |
|---|---:|---:|---:|---|
| history_is | 1701 | 0.693767 | 0.001231 | False |
| interaction | 1701 | 0.010015 | 3.505715 | True |
| interaction | 1702 | 0.012039 | 4.075963 | True |

种子 1702 的确定性交互基线 NLL = 0.011889。因此控制实验支持交互结构的重要性，不支持粒子的必要性。

## 真实开发集实验

同一缓存：440 个训练候选、91 个开发候选；开发集中 40 个可见历史非恒定。均训练 1000 步，训练/评估为 8 粒子，种子 1701，256 维词法哈希特征。以下均为开发集结果，不能与旧 pilot 的测试集 NLL 直接比较。

| 版本 | 选中步数 | 开发 NLL ↓ | shuffle gap ↑ | 配对顺序效应 | 开发筛选 |
|---|---:|---:|---:|---:|---|
| legacy | 250 | 0.524077 | 0.003154 | 0.016939 | False |
| objective | 250 | 0.544154 | 0.017814 | 0.016722 | False |
| interaction | 350 | 0.472410 | -0.000240 | 0.000000 | False |

共同基线（开发集选择后的 NLL）：
- deep_sets: 0.465323
- deterministic_interaction: 0.481648
- no_particle_bottleneck: 0.488634
- pair_aware: 0.466272
- tuned_dirichlet: 0.470472

新模型三个额外采样复核（NLL / gap）：
- 0.502797 / -0.000905
- 0.494341 / 0.000257
- 0.524246 / 0.000124

新模型没有击败最强确定性基线，也没有 association 信号。额外采样下 NLL 高于用于选模的固定采样结果，说明低粒子预算的 Monte Carlo 选模噪声仍需控制。后续协议应预先固定更大的评估粒子预算或多个随机数重复平均，不能用本开发集挑有利的采样种子。

## 下一阶段设计

优先检验表示与数据是否包含可迁移的触发信息：在新的、预先锁定的开发/测试协议下，比较词法哈希与冻结的代码/任务条件语义特征；先让确定性交互模型证明配对证据有效，再比较粒子对不确定性和测试选择的额外价值。训练时可预先规定多种可见前缀视图，但所有方法使用相同训练视图，且开发/测试总体不得按结果事后筛选。
这些是下一阶段假设，不是本次已验证的改进。当前证据不足以把原项目的停止条件改成通过，或升级为 active testing/repair 成功。

## 验证与产物

- 全仓库测试：723 passed, 1 skipped（可选 TeX/Poppler 论文构建），耗时 776.20 秒。
- 新增测试覆盖 causal prefix、配对顺序不变性、不等权重要性比值、latent MMD、梯度方向、失败选模、开发集产物与禁止覆盖。
- 独立代码审查未发现新推断和交互基线的泄漏或正确性阻塞问题；审查提出的参数记录、设备与基线 checkpoint 产物问题已修复。
- 六次运行均正常退出；所有 manifest/checkpoint/result/source 产物按各自 checksums.json 核验。最早的 history-is 可控原型未保存确定性基线权重；第二次可控运行已保存原有三种基线，但尚未加入确定性交互基线。后续运行包含完整的新基线产物。
- 原始日志、checkpoint、逐步指标及源码快照：`/root/pbpf-runs/association-debug-20260916/`。
- 仓库汇总：`results/association_debug_2026-09-16.json`。
- 复现说明：`docs/DIAGNOSTIC_DEBUG.md`。
- 工作分支：`debug/association-inference`，工作树：`/root/PBPF-debug`。
