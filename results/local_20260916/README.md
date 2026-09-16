# 2026-09-16 本地 PBPF 实验结果

本次本机可执行实验已完成；正式 S0–S4 尚未运行。结果基于 `64b603279740fb716107c37c9911ad80be748801` 加上 [运行时补丁](local/compatibility.patch)，不代表随后加入的 APBPF 实验已经运行。

完整记录：[实验执行清单](local/EXPERIMENTS_EXECUTED.md)、[复现报告](local/REPORT.md)、[完成状态](local/completion.json)、[详细结果与阻塞清单](local/additional_completion.json)。

| 实验 | 完成范围 | 结果 |
|---|---|---|
| EvalPlus frozen | 164 任务，每题 4 轮 | 最终通过 86/164（52.44%） |
| EvalPlus repair | 164 任务，每题 4 轮 | 最终通过 72/164（43.90%） |
| RunBugRun 预测 | 3 个种子，每种子 1000 步 | PBPF NLL 0.5299 ± 0.0068；完整因果 Gate 未稳定通过 |
| RunBugRun 修复 | 1500 步训练，8 任务 × 6 条件 | PBPF 1/8，no_latent 1/8，heuristic 2/8 |
| 有限状态独立实验 | 50,000 / 5,000 / 5,000 样本 | test Gate 未通过 |
| 本地合成流程 | 15 阶段及汇总、验证、打包 | 完成，仅验证流程 |
| 代码测试 | 本次运行代码 | 674 passed |

EvalPlus 两组属于历史启发式诊断，不能作为 learned PBPF 有效性的证据。各组分成三个独立批次，Torch 种子均为 0；运行命令和分片信息在各自的 `shards.json`。正式 S0–S4 的 19 个矩阵单元、31 项依附消融，以及未实现的全方法矩阵仍未执行；实际预检查 exit 2，详见 [启动日志](local/logs/formal_launch_all_experiments.log)。

## 文件与复现

- `local/`：原始报告、机器可读汇总、运行脚本、依赖锁、模型校验值、训练和验收日志；保留原始路径及内容。
- `evalplus-task-artifacts.tar.gz`：两组全部 328 个逐任务归档，包含生成程序、事件、评估、配置和 manifest，以及分片汇总。
- `local-diagnostic-artifacts.tar.gz`：exact smoke、RunBugRun 6581 pilot 和生成上限修复后的本地合成归档。
- `SHA256SUMS`：本次发布文件的 SHA256，可在本目录执行 `sha256sum -c SHA256SUMS`。

两个压缩包均以 `local/` 为顶层路径。可在新的空目录中解压查看，避免覆盖已有实验。原始 JSON 中的绝对路径只记录运行机器上的位置，不是下载链接。原始报告中的 `results/...` 和 `logs/...` 路径相对于其所在 `local/` 目录；逐任务目录需从压缩包解压。

复现应从上述原始提交创建独立检出，应用 `local/compatibility.patch`，将本包的 `local/` 复制到该检出；运行脚本记录了原机器路径、缓存和 GPU 布局，需要按目标机器配置。归档内的脚本是实际执行版本，不声称可在任意环境直接运行。测试日志的 674 项仅对应该实验代码版本，不是上传时远端最新 APBPF 代码的测试结果。

本包未包含基础模型权重、训练 checkpoint、下载缓存、原始数据下载目录、有限状态的 120 个 NPZ 数组及中断运行归档。有限状态完整指标、各 NPZ 的校验值与生成脚本已提供；这些大文件保留在原运行目录 `/data/cwj/PBPF/local/`。包含本次 RunBugRun 准备后的候选数据 JSON，以记录实际数据划分。
