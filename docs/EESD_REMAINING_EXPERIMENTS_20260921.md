# EESD 后续实验与 GitHub 交接清单（2026-09-21）

状态快照：**2026-09-21T06:16:02.308900+00:00**。这是全部后续任务的交接清单，不是全部实验完成声明，也不是一键运行器。
机器可读版本：[eesd_handoff_tasks_20260921.json](../configs/experiments/eesd_handoff_tasks_20260921.json)。其中有每阶段依赖、脚本、参数模板、输入输出、阻塞及实际 seal 路径/SHA。`<...>` 是必须从对应锁或前置产物填写的参数，不能原样执行。

## 1. 总范围与计数口径

- 四家族：`qwen25_7b`、`deepseek_6p7b`、`seed_coder_8b`、`starcoder2_15b`。
- 四机制域：RunBugRun、CodeARC、APPS-Replay、CodeContests-Replay。后两域是新增 synthesis/execution-outcome 扩展，不是官方 benchmark pass@1。
- 三个固定生成 seed：1701、1702、1703。
- 总计 **4×4×3=48 个机制 domain/model/seed run**；每个 run 包含 200 development + 500 primary，共 **96 个 split bank、33,600 个候选、336,000 次十测试执行**。
- 49 contrast slots 的数学定义不变。原两域和 Replay 两域分别使用 **8 primary / 1,560 secondary** 家族；不同执行 profile 的产物不混合，不声称跨全部组的联合 .05 FWER。
- SFT 不自动扩展到四模型四数据集。当前 manifest 只有三个 correction cells：Qwen/RunBugRun、DeepSeek/RunBugRun、Qwen/CodeARC。
- 单轮为 **3 cells × 8 arms × 3 training seeds = 72 个计划 policy outcomes**，其中 **63 次计划 GPU update**，另 9 个 no-update baseline。合法零正权重可能减少实际 update 次数，不能删其分析行。

## 2. 当前文件证据（运行会继续变化）

本次仅对以下完成产物重核 run/record 或 direct 完整文件 SHA；没有重跑评分或统计推断。数量按 seal 区分，不把运行中的目录算完成。

| 阶段 | 快照结果 | 总计划/说明 |
|---|---:|---|
| 原两域生成 split banks | 24 已封存 | 48；等于 12 个两 split 齐全的 run |
| Replay 生成 split banks | 10 已封存 | 48；仅 4 个两 split 齐全的 run，其余已封存 split 不等于完整 run |
| 原两域 direct cache+单 seed evidence | 12 已封存 | 24；当前是 Qwen/DeepSeek × 两域 × 三 seed |
| Replay 评分 cache | 0 | 24，尚需 direct backend 适配 |
| direct public correction bank / correction generation receipt | 0 / 0 | 三 correction cells 尚未完成 |
| training report / zero-weight outcome receipt | 0 / 0 | 尚不能声称已训练 |
| Qwen RunBugRun round1 原始训练候选 | train 321/321 已封存；development 120/400 有 checksum、未封存 | 该 development 的 400 不等于机制 development 的 200 |

Qwen 的 correction 原始 banks 在 `runs/eesd-direct-20260921/correction-original-banks/runbugrun/qwen25_7b/round1/`。
完整 seal 列表和 SHA 在机器清单的 `evidence`；未完成候选数只是瞬时进度，不能作为可训练产物。

## 3. 机制、消融和统计的剩余工作

1. 按现有原始 manifest / Replay generation manifest 完成全部 48 runs，保留 Seed 和 StarCoder 的计划行。继续当前生成队列，不重复启动已有作业。
2. 原域采用 `scripts/run_eesd_direct_mechanism_cell.py`，输入原生成 bank、配置和 `runs/eesd-setup/direct-execution-lock-20260921.json`；输出 `runs/eesd-direct-20260921/mechanism/<dataset>/<family>/seed<seed>/`。此入口只接受 `rbr/codearc`。
3. Replay 现有 `build_eesd_replay_mechanism_cache.py`、`run_eesd_replay_evidence.py` 仍绑定原 namespace profile。**不能把它们的旧锁改名成 direct 锁**。需独立 direct adapter、cache/provenance seal，再复用数学与报告。
4. 原域新 `report_eesd_direct_inference.py` 已实现：source closure / 新统计绑定 API 为 `build_statistical_lock(original_statistical_lock, original_sha256, execution_lock, execution_sha256)`；先封存新锁，使用 `--preflight-only` 检查完整 8 cells × 3 seeds。未完成时报告全部缺项，不能解读部分结果为总体结论。
5. Replay 使用独立 8/1560 家族；其 direct provenance/report 适配仍未完成。原 `report_eesd_replay_inference.py` 不应不加说明地接受 direct cache。
6. 每组先封存全部 24 个 run 的 A9 公共特征支持，再重建 assessment。A2/A3、alpha×strength、global/同 kernel mass、均值/打乱 mass、history1/2/4/8、浓度分箱和 binary 均须保留。A8/n4 保持 alias；A7 平均的是指标，不是先平均概率。
7. 10,000 次抽样给出的最小 p 值无法越过 1,560 比较的首个 Holm 门槛；保留这个功效限制，不把无法拒绝说成无效。

参考：[主实验合同](EESD_ICLR2027_EXPERIMENTS.md)、[Replay 扩展](EESD_REPLAY_EXTENSION_SPEC_20260920.md)、[执行变更](EESD_EXECUTION_REVIEW_20260921.md)、`runs/eesd-setup/manifest.yaml`、`runs/eesd-setup/replay-generation-manifest-20260920.json`、`runs/eesd-setup/replay-cache-manifest-20260920.json`。

## 4. 新增 diversity：保留探索性定位

`scripts/report_eesd_diversity.py` 已提供原域 primary-only 三 seed 的 code/AST/outcome 多样性统计，输入原生成 banks 和三个 `--direct-cell SEED=PATH`。必须三 seed 来源总体一致。它是补充探索性分析，不是新增未声明的确认性检验，不选择“表现好”的 seed。

每个原域/模型三份 direct cell 完成后可并行做 CPU 汇总。Replay 尚未有对应 domain/direct adapter；如扩展也须显式记录，不能假定现有原域 CLI 支持。最终成本/生成数量记录同时使用 `summarize_eesd_generation_costs.py`，不能用理论 token 上限冒充实际使用量。

## 5. 三个 correction cells → 单轮训练 72 outcomes

依赖顺序：**完整 train/development 原始 banks → 公共失败机会 → correction generation → score → token-budget plan → 七 update arms → 八臂 fresh evaluation**。

### 可复用的当前入口

- `prepare_eesd_direct_corrections.py`：原参数加 `--execution-lock` / `--execution-lock-sha256`，输入两个完整 bank；输出 `public-corrections.jsonl`、`audit.json`、`direct-profile.json`。
- `generate_eesd_direct_corrections.py`：`--public-bank .../public-corrections.jsonl --model-config <pinned YAML> --domain <rbr|codearc> --output <new directory>`，同样传两个 lock 参数；输出轨迹及 direct receipt。原 correction 的 NF4、greedy 和科学设置不变。
- `score_eesd_corrections.py --input <corrections.jsonl> --config configs/experiments/eesd_downstream_prospective_20260920.yaml --output <new scored directory> --alpha 0.1 --uncertainty-penalty 0.5`，输出 `scored-corrections.jsonl` 和 **`summary.json`**。
- `plan_eesd_training_budget.py --input <scored-corrections.jsonl> --scoring-summary <summary.json> --model-config <pinned YAML> --output <new budget JSON> --seeds 1701 1702 1703`。按 equal-weight 200 optimizer steps、16 microbatches 的参考规则得出每 cell/seed 实际 response-token budget，在训练前封存。此文档不虚构预算数值。
- `run_eesd_downstream.py --stage train --config configs/experiments/eesd_downstream_prospective_20260920.yaml --manifest <one-cell manifest> --output <downstream root> --seed <seed> --response-token-budget <sealed budget>`。每 cell/seed 使用其预算，不能把不同 cell 的预算随意合并。

目录必须分开：原 correction generation 放 `generated-corrections/<dataset>/<family>/round1`；scorer 创建 `corrections/<dataset>/<family>/round1`。trainer 固定从后者读取。scorer 是 create-once，不能把生成输出目录直接作为评分输出重复创建。

CLI 八规则：`no_update`、`equal_weight`、`final_correctness`、`scalar_confidence`、`fixed_mass_dirichlet`、`eed_mean_no_uncertainty`、`eed_no_anchor`、`eesd_full`。对应论文中的等权、正确性筛选、标量置信度、固定证据量、无不确定性项、无 KL anchor 和完整 EESD。

训练前仍需 anchored SFT 显存可行性检查。训练 binding 必须保留 scored input、model、seed、rule、预算、parent 和 trainer SHA；不能接受未绑定旧 adapter。

[零权重采用协议](EESD_ZERO_WEIGHT_ADOPTION_20260920.md) 已实现：合法零正训练权重记 `non_estimable_zero_positive_training_weight`、实际0 token、无 adapter；主分析保留缺估计行，不伪造等预算训练、不启用 unchanged-policy fallback。

### 还未 direct 一键连通的部分

- `--stage fresh` 规划八臂已实现，但默认调用旧 `evaluate_eesd_fresh_bank.py` 的隔离执行；需独立 direct evaluator/profile binding。
- correction credibility 的 `report_eesd_correction_credibility.py --evaluator-root` 仍执行旧 backend；应新增 direct 评估适配，或读取经过正确 direct seal 绑定的独立 `--evaluation-dir`。隐藏评估结果只用于可信度报告，不能用于训练样本选择。
- CPU准备、训练、fresh测试可流水并行，但仅在对应上游 seal 完整且资源允许时；本清单不自动启动它们。

## 6. Qwen 双域 round0–3 递归

主协议是 [shared EESD teacher 锁](EESD_RECURSIVE_SHARED_TEACHER_LOCK_20260920.md)：两个域、三个 seed；每轮从同一个进入轮次的 teacher、共享经验 bank 和相同 token budget 分叉 `equal_weight` / `eesd_full`。三次更新对应 **36 个计划 GPU updates**（零权重例外另记），round0 是基础策略评估。

EESD 固定成为下一轮 teacher；equal-weight 是该 teacher 的单轮 matched control，不是自己的独立递归链。比较包含 teacher-relative 和 vs-base，不能把 control 相邻显示轮次当自身持续学习。

现有 `run_eesd_recursive.py` / downstream `--stage recursive --experience-policy shared_eesd_teacher --response-token-budget <sealed budget>` 已实现协议与 resume receipts，但内部 subprocess 仍指向原 prepare/generate/fresh evaluator。**尚需 direct orchestration adapter**，不能把实现了科学流程等同于当前环境已可直接运行。继续 teacher 若为空，则后续轮 `blocked_dependency`；独立其他 cell/arm 继续。

## 7. HumanEval+ / MBPP+ 迁移

必需范围：Qwen 和 DeepSeek 的 RunBugRun 训练策略，迁移至 HumanEval+、MBPP+。当前 transfer 路由为五控制臂 `no_update/equal_weight/final_correctness/fixed_mass_dirichlet/eesd_full`，三 seed，共 **60 个计划 policy/seed/benchmark outcomes**；不是把所有八训练臂自动扩成迁移任务。

公开生成入口 `run_eesd_evalplus_transfer.py` / downstream `--stage transfer` 已接受 `--public-data-root runs/eesd-data/evalplus-locked/public --public-manifest-sha256 <actual SHA>`；模型只能读公开 prompt/entry point。已有版本锁：EvalPlus0.3.1、HumanEval v0.1.10、MBPP v0.2.0，private 原始数据独立保留。

官方 sanitizer/evaluator 现有入口 `evaluate_eesd_evalplus_isolated.py` 仍要求 bwrap。用户已选择 direct 执行后，需单独实现、封存 **官方 EvalPlus 的 direct profile**，保留 Base 与 Base+Extra 的官方评分规则和每 task coverage。当前生成 CLI 的 `--evaluate` 不提供隐式裸跑 fallback；不能仅开该开关宣称迁移已完成。训练零权重策略仍保留缺 policy 行。

## 8. 可选项目不丢失

- **LiveCodeBench release_v6**：六 JSONL 已下载并有官方 SHA receipt，1,055 rows；这是后续可选 temporal transfer，不是四机制域之一。`materialize_eesd_livecodebench.py` 可用，但独立数据/时间窗/执行协议及实际迁移仍待完成。不要临时反序列化 private pickle 或执行远端 builder。
- **SWE-bench Lite**：主合同的可选 repository appendix；当前不在必需机制/训练计数内，未排入正式队列。
- 保留全因子、逐 seed、成本、历史诊断及负结果附录。辅助测试通过不计科研完成。

## 9. GitHub / 大文件备份交接边界

这两份新文件将未来所有阶段列入交接，但**不证明已 push，也不证明模型、数据、adapter 等大文件已上传 GitHub**。root 负责实际 commit/push，并记录 remote/commit 和成功结果。大文件应有独立存储副本及清单/SHA；代码仓库至少保留 pinned revisions、下载 receipts、各阶段 locks、运行参数、结果 receipts 和恢复方法。

恢复顺序：核对 repo commit与源码锁 → 恢复/核验大文件 → 读取机器清单快照并重扫当前 seals → 恢复未完成生成 → direct评分与CPU准备并行 → 预算封存后训练 → fresh/recursive/transfer direct适配与评估 → 完整家族报告。不要覆盖已有产物或因为效果负面丢弃 cell。
