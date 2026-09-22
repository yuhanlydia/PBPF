# 四张 GPU 每卡 5 个任务实施计划

## 最新核验（2026-09-22 01:22 UTC）

12/12 个 `no_update` 基线均已在每组 500 题上封存。84 个方法任务已准入：11 个完成训练，3 个因无正权重样本标为不可估计，6 个正在训练，64 个待启动。独立评估已完成 9 组，每组 500 题；APPS Replay × Gemma 3 4B 与 RunBugRun × Gemma 3 4B 的各五个新候选分片正在运行。其余可估计方法仍随训练结果滚动接入。实时进度以 [`seed1701-live-snapshot.json`](../../../runs/eesd-direct-20260921/operations/seed1701-live-snapshot.json)、[`seed1701-training-queue.json`](../../../runs/eesd-direct-20260921/operations/seed1701-training-queue.json) 与 [`seed1701-fresh-evaluations.json`](../../../runs/eesd-direct-20260921/operations/seed1701-fresh-evaluations.json) 为准。

APPS Replay × Qwen2.5 Coder 7B × `eesd_full` 的 500 个新候选已封存并完成独立评估。其基线／主方法隐藏测试通过为 `67/500`／`69/500`，全测试通过均为 `57/500`；两份报告的任务清单摘要一致。首次汇总因评估器按旧版顶层字段读取 `public_tasks_sha256` 而失败；已改为兼容新版 `bindings` 字段，保留失败收据与日志，复用已封存候选完成评分。定向测试 3 项通过；全量测试为 1494 通过、13 失败、1 跳过，失败包括当前主机不支持旧沙箱命名空间以及旧注册表与新增模型配置不一致，不能据此声称全库通过。四卡调度目标仍为每卡 5 个独立任务，实际补位受显存准入约束。

CodeARC × DeepSeek 6.7B × `eesd_full` 已按 `201321/201321` 响应 token 完成训练，共 210 个优化步骤；训练绑定收据已落盘。GPU 1 自动补入 CodeContests Replay × DeepSeek 6.7B × `eesd_full`，独立评估队列同时启动 CodeARC × DeepSeek 的五个候选分片。

CodeARC × DeepSeek 6.7B × `eesd_full` 的五个候选分片和 500 题独立评分现已完成。与同任务清单的 `no_update` 基线相比，隐藏测试通过从 `79/500` 到 `110/500`，全测试通过从 `75/500` 到 `102/500`；报告清单摘要一致，评估报告 SHA-256 已写入队列收据。CodeContests Replay × DeepSeek 6.7B × `eesd_full` 已写出 `step-000010` 可恢复断点。

APPS Replay × Gemma 3 4B × `eesd_full` 已按 `1061130/1061130` 响应 token 完成训练，共 299 个优化步骤；训练绑定已封存。其五个独立候选分片在 GPU 0、1、3 上运行。GPU 0 同时自动补入 CodeContests Replay × Gemma 3 4B × `eesd_full` 训练。

RunBugRun × Gemma 3 4B × `eesd_full` 已按 `756554/756554` 响应 token 完成训练，共 188 个优化步骤；绑定收据已封存。其五个独立候选分片已经启动，GPU 2 自动补入 RunBugRun × Qwen2.5 Coder 7B × `eed_mean_no_uncertainty` 训练。

## 历史执行快照（2026-09-21 22:59 UTC）

本节记录正在执行的 seed 1701 主实验；下方的“五槽具体排法”和迁移实施清单是早期预案，不能当作当前进程表。实时事实见 [`seed1701-live-snapshot.json`](../../../runs/eesd-direct-20260921/operations/seed1701-live-snapshot.json)，训练与独立评估的权威队列收据分别是 [`seed1701-training-queue.json`](../../../runs/eesd-direct-20260921/operations/seed1701-training-queue.json) 和 [`seed1701-fresh-evaluations.json`](../../../runs/eesd-direct-20260921/operations/seed1701-fresh-evaluations.json)。

| 阶段 | 当前结果 | 后续动作 |
|---|---|---|
| `no_update` 基线 | 4 benchmark × 3 模型的 12 个 500 来源评估已封存 | 保留配对评估清单 |
| 修正数据与预算 | 12/12 单元已封存并生成训练计划；最后完成的 RunBugRun × DeepSeek 6.7B 与 RunBugRun × Gemma 4B 各有 7 个合格方法，预算分别为每方法 `725074` 与 `756554` 响应 token | 保留封存输入、评分与预算收据 |
| 方法训练 | 84/84 个方法任务已准入；6 个运行、68 个待启动、7 个已按精确预算完成、3 个科学意义上的“不可估计”已封存 | 显存允许时由队列补位；新计划使用可恢复的 adapter＋优化器断点 |
| 独立评估 | 6 个已训方法完成 5 × 100 来源新候选与独立评分；CodeARC × Gemma 主方法的 5 个分片正在运行；另有 3 个“不可估计”单元 | 其余训练方法完成后滚动接入 |
| 结果表 | 12 个基线齐全；6 个方法有独立评分，3 个方法不可估计，尚缺 75 个方法结果 | 输出主结果表和七方法对照表；保持 benchmark 内同模型、同任务、同 seed 的配对比较 |

RunBugRun × Qwen2.5 Coder 7B × `eesd_full` 已按 `718289/718289` 响应 token 完成训练，并在同一 500 来源任务清单上完成独立评估。`no_update` 基线与主方法的隐藏测试通过分别为 `239/500`、`248/500`，全测试通过分别为 `228/500`、`240/500`；两份报告的任务清单摘要一致。CodeARC × Gemma 3 4B × `eesd_full` 已按 `269738/269738` 响应 token 完成训练，其独立评估正在生成。最新进程、断点与分片状态以本节链接的队列收据为准。

首个已训模型为 CodeARC × Qwen2.5 Coder 7B × `eed_mean_no_uncertainty`，训练报告确认精确消耗 `215549/215549` 个响应 token、231 个优化步骤，模型绑定收据已封存。该方法的独立评估为隐藏测试通过 `90/500`，与同范围 `no_update` 基线的 `90/500` 持平；全测试通过分别为 `86/500` 和 `85/500`。两份报告的任务清单摘要一致。四卡利用率和实际任务数随训练与分片交接变化，以实时快照和队列收据为准。`5/5/5/5` 仍是每卡调度目标，只有真实独立任务且显存满足准入阈值时才补位。训练与新候选生成共享 GPU 启动锁；进程启动后的 120 秒保留显存预留。Qwen CodeARC 的首批 5 个训练任务仍使用原始训练脚本，其后计划使用带断点恢复的训练脚本；两者的源码摘要在各自计划中固定，不混用。

CodeARC × Qwen2.5 Coder 7B × `equal_weight` 也已完成精确 `215549/215549` token 训练和独立评估；隐藏测试通过 `99/500`，全测试通过 `94/500`，与同任务范围的 `no_update` 基线 `90/500`、`85/500` 比较。CodeARC × Gemma 3 4B × `eesd_full` 已运行带断点训练脚本；`step-000020` 断点包含 adapter、优化器及随机数状态，`state.pt` 和 adapter 树的 SHA-256 均与 `latest.json` 匹配。

CodeARC × Qwen2.5 Coder 7B × `final_correctness` 已完成精确 `215549/215549` token 训练，共 254 个优化步骤，训练报告 SHA-256 与绑定收据匹配；独立评估的隐藏测试通过 `97/500`，全测试通过 `90/500`，评估报告 SHA-256 与队列收据匹配。Gemma 训练的断点继续滚动更新，实时步骤以 `checkpoints/latest.json` 为准。

CodeARC × Qwen2.5 Coder 7B × `eesd_full` 和 `eed_no_anchor` 也已完成精确 `215549/215549` token 训练，报告与绑定 SHA-256 匹配并进入独立评估。CodeContests Replay × Gemma 4B 的 `final_correctness` 因训练权重无正样本被明确标为不可估计，不按 0 分计入。

其中 `eesd_full` 的独立评估已完成：隐藏测试通过 `87/500`，全测试通过 `83/500`；同任务范围的 `no_update` 基线分别为 `90/500` 和 `85/500`。评估报告 SHA-256 与队列收据匹配。GPU 空位已自动补入 CodeARC × DeepSeek 6.7B 和 APPS Replay × Qwen2.5 Coder 7B 的 `eesd_full` 训练。

`eed_no_anchor` 的独立评估也已完成：隐藏测试通过 `91/500`，全测试通过 `86/500`，报告 SHA-256 与队列收据匹配。APPS Replay × DeepSeek 6.7B 的 `eesd_full` 已自动补入训练队列，当前共有五项方法训练并行运行。

**接下来按单元滚动推进：** 已准入的七方法训练 → 每方法独立评估 → 主结果表与七方法对照表。最终必须核齐 84 个方法单元和 12 个 `no_update` 基线；“不可估计”保留为明确状态，不填作 0 分。最终结果由 [`render_eesd_seed1701_results.py`](../../../scripts/render_eesd_seed1701_results.py) 生成。下方早期分片实施项的复选框只记录当时的迁移设计，不代表当前阶段的完成比例。

> **执行方式：** 按下列任务逐项实现、验证和上线；每项结束后记录命令、结果与回滚点。

**目标：** 仅用 seed 1701，将剩余工作切成互不重叠的独立任务；四张 A6000 **每张同时运行 5 个实际 GPU 任务**，任务结束后立即从同一 seed 的就绪队列补位。

**约束：** 只运行 seed 1701，绝不为补位启动 1702／1703 或新 seed。模型仅限 Gemma 3 4B Instruct、DeepSeek Coder 6.7B Instruct、Qwen2.5 Coder 7B Instruct。每张卡的五个任务必须拥有不同、互不重叠的任务 ID 或不同的已准入方法工作。已有结果与断点必须保留。不得以空转进程、重复写者或未满足输入门槛的评分／训练凑数。

**早期迁移事实（历史）：** seed 1701 的 Gemma 在四个 benchmark 各有一个生成写者；DeepSeek 的 CodeContests Replay seed 1701 仍在跑；Qwen 的该单元已完成。仍在运行的 Qwen／DeepSeek seed 1702／1703 已停止，记录保留。当时只有约五个 seed 1701 写者，尚未达到每卡五个。

**架构：** Gemma 的四个 benchmark 各有固定的 700 个 seed 1701 任务 ID（200 development、500 primary）。先停写者并校验断点，再将各 benchmark 未完成的 ID 确定性地分给独立分片。每个分片有自己的任务身份、输出目录和锁。统一调度器按下面的每卡五槽表派发；每 15 秒核对实际 GPU 进程、显存、结果封存和空位。纯候选生成使用预先固定的公开输入，development 与 primary 可同时生成；评分、训练仍要等待各自规定的输入封存。

**技术栈：** Python 3.11、PyTorch、Transformers、bitsandbytes、`nvidia-smi`、现有 EESD 公共数据和校验清单。

## 科学实验矩阵与控制变量

**主实验只有 12 个单元：** `4 个 benchmark × 3 个模型 × seed 1701`。GPU 分片是完成同一个单元的并行执行方式，不能算成新增 benchmark、模型、seed 或独立重复。

| 主 benchmark | Gemma 3 4B | DeepSeek Coder 6.7B | Qwen2.5 Coder 7B |
|---|---|---|---|
| RunBugRun | seed 1701 | seed 1701 | seed 1701 |
| CodeARC | seed 1701 | seed 1701 | seed 1701 |
| APPS Replay | seed 1701 | seed 1701 | seed 1701 |
| CodeContests Replay | seed 1701 | seed 1701 | seed 1701 |

**比较方向 A：固定 benchmark，跨模型。** 每一行比较三个模型；同一行必须使用同一公开任务 ID、同一 development／primary 划分、同一候选数、同一提示信息范围和同一评分规则。模型本身及其必要的原生 chat template 可以不同，须记录该差异。

**比较方向 B：固定模型，跨 benchmark。** 每一列看同一个模型和同一方法在四个 benchmark 上的表现。各 benchmark 任务难度与生成上限可能不同，所以不直接把原始通过率的大小当成“数据集带来的因果效应”；主要比较同一模型在各 benchmark 内的方法增益、方向及置信区间，并列出每个 benchmark 的任务数与生成预算。

**方法控制：** 同一实验单元内的候选生成、评分和后续训练／评估须引用同一组封存输入；比较方法时固定模型、benchmark、seed、数据划分和预算，只改变被研究的方法。任何分片结果先合并并校验为原来的 200 development＋500 primary，之后才进入评分。

## 每卡五槽的具体排法

| GPU | 槽 1 | 槽 2 | 槽 3 | 槽 4 | 槽 5 |
|---|---|---|---|---|---|
| 0 | Gemma／RunBugRun 分片 A | 分片 B | 分片 C | 分片 D | 分片 E |
| 1 | Gemma／CodeARC 分片 A | 分片 B | 分片 C | 分片 D | 分片 E |
| 2 | DeepSeek／CodeContests Replay seed 1701 | Gemma／APPS Replay 分片 A | 分片 B | 分片 C | 分片 D |
| 3 | Gemma／CodeContests Replay 分片 A | 分片 B | 分片 C | 分片 D | 分片 E |

GPU 2 的 DeepSeek 任务完成后，槽 1 改为 Gemma／APPS Replay 分片 E；若某个 Gemma 分片提前完成，优先接同一 seed 已就绪的评分／方法任务，否则把该 benchmark 尚未生成的 ID 再拆成独立分片。不能拆出非空任务时报告该槽已无有效工作，而不是启动重复任务。所有分片合并后，每个 benchmark 恰好覆盖原来的 700 个 ID，每个 ID 只生成一次。

**轮次 1（现在）：** 保留正在运行的四个 Gemma seed 1701 写者及 DeepSeek／CodeContests Replay seed 1701；不再运行其他 seed。Qwen／CodeContests Replay seed 1701 已完成。

**轮次 2（迁移后）：** 停止并校验四个 Gemma 写者，按上表将各 benchmark 剩余 ID 分片，在四卡启动 `5/5/5/5`。先迁移已有完整记录，再对未完成 ID 采样；不改 Gemma 模型、提示内容、任务 seed 或候选预算。

**轮次 3（生成结束后）：** 将 Gemma 分片合并为四个 benchmark 的完整 bank，连同 Qwen／DeepSeek seed 1701 结果补齐上面的 12 单元矩阵，再统一评分与方法比较。每个新评分或训练任务只有输入就绪且能占用 GPU 时才替换完成的生成槽。

## 运行规则

1. **计数：** 对每张 GPU 分别计数，只计实际执行模型生成、评分或训练的活进程。父进程、CPU 预检、等待依赖的任务不计数。迁移后的运行目标始终是 `running_by_gpu = {0:5,1:5,2:5,3:5}`，每卡单独补位。
2. **准入：** 启动前确认模型不超过 7B、任务身份唯一、输入清单和模型权重校验通过、输出目录没有其他写者、依赖已封存、GPU 空闲显存高于该任务实测峰值加安全余量。启动后复核显存；OOM 不得自动无限重试。
3. **补位顺序：** 同一 seed 中已就绪的不同 benchmark／方法任务优先，其后为各卡绑定 benchmark 的未完成 ID 分片。绝不追加 seed。纯候选生成的 development／primary 输入均已固定，可分片并行；评分和训练仍需等待相应 bank 完整封存。
4. **断点：** 仅跳过已有且校验通过的记录；残留 `.partial`、损坏校验和或身份不一致时隔离该任务并报警，不覆盖旧文件。
5. **不可达状态：** 当有效就绪任务少于空位，或安全显存不足时，状态文件必须写明 `target=5`、`actual`、`shortfall_reason` 和阻塞任务。不得宣称已经保持 5 个；任务全部完成后自然降到 0。

## 实施任务

### 1. 冻结任务与资源清单

- [ ] 读取 `/root/eesd-migration-20260921/gemma-generation-status.json`、现有 Qwen／DeepSeek 生成目录和四卡进程表；形成按 `benchmark × model × seed × split × method` 的清单。
- [ ] 为每项记录状态（完成、运行、就绪、依赖阻塞、失败）、输出目录、模型版本、输入摘要与预计峰值显存；排除所有非 1701 seed、8B 和 15B 项。
- [ ] 保存只读快照和 SHA-256；核对当前 Gemma 与 CodeContests Replay 写者没有重复。

### 2. 增加有效并发任务

- [ ] 先安全停止四个 Gemma 写者，逐条验证既有记录和校验和；冻结每个 benchmark 的 700 个 seed 1701 ID 与已完成 ID。停止期间四卡会短暂低于五槽，不能把迁移窗口算作达标运行。
- [ ] 为 Gemma 的剩余公开样本生成源组件级分片清单：RunBugRun 5 份、CodeARC 5 份、APPS Replay 先 4 份后 5 份、CodeContests Replay 5 份。分片 ID 两两不交，与已完成 ID 合并后恰好覆盖原始 700 个 ID；不引入第二个 seed。
- [ ] 在新版本 Gemma 生成器中支持 `--shard-manifest`；每个分片有独立 `run.json`、记录目录、锁和 `complete.json`。保持原始提示内容、采样参数、每任务 seed 与模型版本不变。
- [ ] 迁移已有完整记录到新分片时使用校验后的只读复用或硬链接，并保存来源收据；任何已有任务 ID 只能由一个分片认领。原目录保留，不覆盖。
- [ ] 对四个 Gemma benchmark 各生成一份分片合并清单，校验 200 development＋500 primary、任务 ID 无重无漏、来源与模型版本一致；合并后仍只记一个 seed 1701 实验单元。
- [ ] 预计算每卡五个模型进程的实测显存峰值，确认加安全余量仍低于 48 GiB；否则减小分片并发所需显存或报告不可达，不能冒险 OOM。

### 3. 建立统一调度器

- [ ] 用单实例文件锁启动调度器；启动时按完整命令、PID、进程会话、GPU 环境变量和输出锁接管现有写者。
- [ ] 维护 `running/pending/blocked/complete/failed` 状态；每次循环先验收退出任务，再按优先级和显存准入为每卡补到最多 5 个。
- [ ] 每次只向每卡加载一个新模型，待显存稳定后重新查询 `nvidia-smi`，避免启动瞬间的显存竞态。统计 GPU 子进程，而非只统计包装器。
- [ ] 任务失败时保留断点、记录退出码与末尾日志，按明确的可重试规则处理；身份或校验错误必须人工排查，不能盲目重启。
- [ ] 状态文件至少包含时间、每卡目标／实际数、显存余量、任务键、PID、完成量、失败原因与补位阻塞原因；日志和状态写入采用原子替换。

### 4. 验证与上线

- [ ] 用模拟队列验证上表的 20 个 seed 1701 任务能填满 `5/5/5/5`，每卡单独补位；任何 1702／1703 任务都被拒绝，任务不足或显存不足时准确报告缺口。
- [ ] 验证分片无重复、无遗漏、跨 split 无源组件泄漏，已有 Gemma 记录可按原校验和续跑；验证 8B／15B 永远不准入。
- [ ] 验证最终 12 单元矩阵全部具有相同的 seed 1701 和各 benchmark 内相同的任务 ID／评分协议；生成模型维度与 benchmark 维度两套汇总表。
- [ ] 先在一张卡用两个真实 Gemma 分片做短程试跑，检查每条记录的来源、提示摘要、候选数和校验和；再启用四卡调度。
- [ ] 上线后连续观察至少三次补位循环和一次真实完成／补位事件，保存状态快照。若出现 OOM、写者冲突或校验失败，停止新派发并保留现有有效进程和断点。

## 验收标准

- 有不少于 20 个**已就绪且能通过显存准入**的独立任务时，稳定状态达到 `5/5/5/5`；一个任务结束后自动补入下一个。
- 不足 20 个有效任务时，系统准确报告缺口及原因；不以重复任务或空转进程凑数。
- 全部运行任务均为 seed 1701，模型不超过 7B；已有输出和校验和无覆盖；每个任务 ID 最多一个活写者。
- 最终结果恰好是上述 12 个科学实验单元；GPU 上的 20 个并发分片不被误报为 20 个独立实验。
- 生成结果仍需后续评分和方法训练；进程数本身不代表科学结果已完成。
## 2026-09-22 恢复运行与监督

当前执行以 seed 1701 的 84 个方法训练臂、12 个 no_update 基线和独立 fresh evaluation 为准。训练及评估调度器分别使用文件锁和 GPU 准入锁；`scripts/supervise_seed1701.py` 每 60 秒检查调度器与监控进程，缺失时重启，并把四卡实际进程数写入 `operations/seed1701-supervisor.json`。完整或失败的队列不会被自动重启；失败记录需要排查。

恢复时，六个已经产生完整训练报告、但因调度器退出未封存的训练臂已经重新校验并封存。恢复后的队列为 20/84 完成、4 运行、60 待运行；评估为 11 项完成、3 项不可估计、6 项待评估。四张 RTX A6000 已重新承担训练与评估工作。每卡最多五个任务；显存准入不足时保持较少的真实任务，不能为了凑数启动重复任务。
