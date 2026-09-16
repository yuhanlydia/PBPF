最新实验执行清单：[EXPERIMENTS_EXECUTED.md](EXPERIMENTS_EXECUTED.md)。补跑状态见 `additional_completion.json`，运行中批次见 `evalplus_sharded_status.json`。

# PBPF 本机复现

仓库： https://github.com/yuhanlydia/PBPF
提交： `64b603279740fb716107c37c9911ad80be748801`（main 已包含 formal 分支）

```bash
cd /data/cwj/PBPF
source local/activate.sh
```

运行脚本（在仓库根目录）：

- `bash local/run_smoke.sh`：本地合成数据 15 阶段流程及验证/打包。
- `bash local/run_prediction.sh`：官方 RunBugRun 数据准备、三个随机种子各 1000 步训练及预测评估。
- `python local/download_actor.py`：下载固定版本 Qwen2.5-Coder-7B-Instruct。
- `bash local/run_repair_smoke.sh`：2 步训练、1 个任务的模型加载/反向传播/六组生成检查，不能作为修复效果证据。
- `bash local/run_repair.sh`：冻结 4-bit Qwen，训练 soft prefix 1500 步，评估 8 个候选程序。

默认使用物理 GPU 0，可用 `CUDA_VISIBLE_DEVICES=1 bash local/run_prediction.sh` 指定其它卡。
数据在 `local/data/`，Hugging Face 完整模型缓存入口为 `local/model-cache/`（链接至 SSD）；`local/hf/` 保留下载中间文件，输出在 `local/results/`，日志在 `local/logs/`。
真实数据脚本通过 Bubblewrap 运行，禁用网络；依赖与仓库只读，`local/` 可写。
依赖锁定、实际验证结果和复现边界见 [REPORT.md](REPORT.md)。
环境重建命令：`bash local/setup.sh`。所有测试：`bash local/verify.sh`（边界测试样例确定性修正后为 671 passed；原版失败记录保留，详见报告）。

## 上游边界

`local_cpu` 只是 `smoke-only-no-claim`，不代表论文结果。
仓库没有完整的正式生产科学流水线。H200/Slurm 正式实验需要外部 factory、独立 evaluator、数据和模型快照等部署条件。
上游 RunBugRun 预测报告的完整因果 Gate B 未通过，8 个候选的 repair pilot 也为负结果。
本机复现保留原始算法，不将实验失败修改为成功。

数据来源：

- https://github.com/giganticode/run_bug_run_data/releases/tag/v0.0.1
- https://github.com/IBM/Project_CodeNet/blob/main/doc/problem_descriptions.tar.gz
- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242

注意：重跑预测会覆盖本机同名结果；repair 脚本存在上游 resume 行为，要完全从头运行应先移动旧 projector 权重并保留原结果。

## 修复实验状态

完整实验由 `local/full_repair_job.py` 运行及核验；进度在 `local/repair_status.json`，训练日志为 `local/logs/repair.log`。只有 `full_1500_step_pilot_completed` 为 true 才表示 1500 步训练和 8 任务、6 组评估均已完成。
