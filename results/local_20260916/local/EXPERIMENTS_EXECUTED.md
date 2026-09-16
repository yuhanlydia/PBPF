# 仓库实验执行清单

目标是执行文档列出的实验，不以复现作者的具体数字为完成标准。

## 已完成

| 实验 | 执行范围 | 结果文件 |
|---|---|---|
| ICLR 本地合成 DAG | 15 阶段及 aggregate / verify / package | `results/iclr_verified_after_generation_cap/` |
| RunBugRun 预测 | 3 个种子 × 1000 步 | `results/rbr_gate_b_aggregate.json` |
| RunBugRun 冻结 Qwen 修复 | 1500 步，8 任务 × 6 组 | `results/rbr_repair_gate.json` |
| 原始 exact smoke | 4 个修复回合，归档校验通过 | `results/legacy_exact_smoke/` |
| 原始 RunBugRun 单任务 pilot | 6581 号任务，4 个修复回合，归档校验通过 | `results/legacy_rbr_6581/` |
| 候选库选择审计 | 原始 GOAV 164 组 × 8 候选，输入 SHA256 与仓库一致 | `results/selector_bank_audit.json` |
| 有限状态配置的独立运行 | 50,000 / 5,000 / 5,000 样本，10 组方法 | `results/finite_contract/summary.json` |
| EvalPlus frozen 16GB | 164 任务 × 4 回合，164 个归档校验通过；最终通过 86/164 | `results/evalplus_frozen_164/` |
| EvalPlus repair 24GB | 164 任务 × 4 回合，164 个归档校验通过；最终通过 72/164 | `results/evalplus_repair_164/` |

最终代码测试 674 项通过。官方标准答案的执行正对照：RunBugRun 80/80，HumanEval 164/164。七个 legacy 配置的 plan 校验全部完成；plan 校验不代表实际执行对应的全部方法。

有限状态运行使用 `local/run_finite_contract.py` 补充数据生成驱动，调用原仓库 `pbpf.finite.run_finite_audit`。随机流的字节编码在结果中显式记录。它是按配置完成的独立本地实验，不是正式 factory 产生的 S1 归档；测试集 Gate 未通过，未调整数据或门槛。

EvalPlus 使用文档已有入口，将每组 164 任务分成 0–54、55–109、110–163 三批。每批使用原脚本的 Torch 种子 0，模型、生成参数、8 候选和 4 回合保持配置值。已修复原 ConditionedRepairBackend 将单次 192 token 上限覆盖为剩余总预算的问题，并以修复后的代码重新运行全部批次。批次命令见各结果目录的 `shards.json`。两组参数分别实际运行，不共用生成结果。

## 无法执行的文档项目

正式 S0–S4 的 19 个矩阵单元及依附该流程的 31 项消融：正式启动命令已实际执行，exit 2，未启动模型或作业。仓库没有提供完整 production FormalFactory；本机也没有 PBPF_ICLR_SITE、独立 evaluator authority、正式快照/容器和经配置的 H200 资源。配置中的 frozen_7b_16gb、repair_7b_24gb、formal_h200 全方法矩阵同样只提供预注册配置，没有完整执行入口。

这属于缺少可执行实现和部署输入，与是否要求匹配作者指标无关。没有用合成输出、旧诊断或仅通过 plan 校验来标记这些实验完成。全部未执行单元及具体配置列于 `additional_completion.json`，实际错误见 `logs/formal_launch_all_experiments.log`。

候选选择的更大、更难新候选库是文档提出的后续研究计划，仓库没有提供新候选库和完整训练/评估入口；已执行文档现有的原始候选库审计。

## 运行与保存

工作目录 `/data/cwj/PBPF`，环境 `.venv`。模型缓存现通过 `local/model-cache` 指向服务器迁移后的 `/data/cwj/.cache/pbpf-huggingface-root123`；沙箱动态挂载真实路径。系统盘满时，新增实验临时输出使用 `/dev/shm/pbpf-additional`，已完成结果复制到本项目 `local/results/`。

新命令：
```bash
source local/activate.sh
python local/summarize_additional.py
python local/write_execution_report.py
```

原有训练、预测及本地 DAG 的命令和详细指标见 `REPORT.md`；本清单覆盖后来补跑的实验。
