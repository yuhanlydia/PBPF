"""Render the final local reproduction report from measured artifacts."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1];local=root/'local'
def read(name): return json.loads((local/name).read_text())
env=read('environment.json');prediction=read('results/rbr_gate_b_aggregate.json');status=read('repair_status.json')
pilot_path=local/'results/rbr_repair_gate.json';pilot=json.loads(pilot_path.read_text()) if pilot_path.exists() else None
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
lines=['# PBPF 本机复现报告', '',f'代码：`{root}`；上游提交：`{commit}`。', '',
'## 完成范围','',
'- 独立 Python 环境、CUDA 和实验依赖安装完成，依赖锁见 `requirements.lock.txt`。',
'- 本地合成 15 阶段流程及 aggregate / verify / package 完成；代码兼容修正后重新验证的归档位于 `results/iclr_verified_after_compat/`。',
'- RunBugRun 真实数据预测：1701、1702、1703 三个种子，每个 1000 步，全部完成。',
f'- 2 步、1 任务的修复集成检查：{"完成" if status.get("repair_smoke_completed") else "未完成"}。',
f'- 1500 步训练、8 任务、6 组对照的完整本地修复诊断：{"完成" if status.get("full_1500_step_pilot_completed") else "未完成"}。',
'- 测试：原版 670 passed / 1 failed；确定性修正测试样例后 671 passed。没有跳过失败项。',
'', '## 环境', '', f'Python {env["python"].split()[0]}；Torch {env["torch"]}；CUDA {env["cuda"]}；Transformers {env["transformers"]}；bitsandbytes {env["bitsandbytes"]}；RTX 4090。',
'', '```bash','cd /data/cwj/PBPF','source local/activate.sh','```','',
'完整模型缓存入口为 `local/model-cache`，实际位于 SSD `/home/root123/.cache/pbpf/huggingface`。仓库及实验结果入口位于 `/data/cwj/PBPF`。',
'', '## 真实数据预测', '',
'官方 RunBugRun v0.0.1 的五个源文件 MD5 和 CodeNet 描述归档 SHA256 均与源码预期一致。当前默认准备参数是 160/32/96 个问题上限、每题最多 4 个候选、每个程序 10 个测试。',
f'过滤后候选数：{prediction["records"]}；问题数：{prediction["problems"]}；测试集未来结果数：{prediction["future_examples"]}。四个测试可见、六个用于未来结果评估。',
'', '| 条件 | NLL，均值 ± 样本标准差 | 准确率 |','|---|---:|---:|']
labels={'baseline':'无证据','history_rate':'可见结果频率基线','pbpf':'PBPF','shuffled':'打乱关联','wrong_candidate':'错候选证据','random':'随机 latent'}
for arm in labels:
 m=prediction['metrics'][arm]
 lines.append(f'| {labels[arm]} | {m["nll"]["mean"]:.4f} ± {m["nll"]["sample_std"]:.4f} | {m["accuracy"]["mean"]:.2%} |')
lines += ['', '三个种子均降低相对无证据和频率基线的 NLL，但打乱关联仍保留收益。full_gate_passes 分别为 false / true / false；不能称为通过完整因果 Gate B。',
'', '## 冻结 Qwen 的实际代码修复', '',
'固定模型 Qwen/Qwen2.5-Coder-7B-Instruct，revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`，4-bit。训练 1500 步、学习率 3e-5、序列上限 1024、8 个 prefix token、prefix RMS 0.02、delta RMS 0.002；每 250 步在 development 上选择权重。评估 8 个相同 test 候选、6 组条件、greedy、最多 512 个新 token。',
'完整训练从头开始，未采用 2 步 smoke 权重。Python/NumPy/Torch 的初始化种子均固定为 2701。']
if pilot and status.get('full_1500_step_pilot_completed'):
 lines += ['', f'训练样本 {status["training_items"]}；按 development NLL 选中第 {status["selected_step"]} 步。', '', '| 条件 | 全部测试通过的候选 | 未来测试通过比例 |','|---|---:|---:|']
 for arm, m in pilot['summary'].items(): lines.append(f'| {arm} | {m["solved"]}/{m["tasks"]} | {m["future_pass_fraction"]:.2%} |')
 lines += ['', '原始生成代码、每个测试的结果和生成耗时保存在 `results/rbr_repair_gate.json`；完整训练记录在 `logs/repair.log`，权重为 `results/rbr_soft_prefix.pt`。',
 '官方 fixed programs 的独立执行正对照为 80/80 PASS，说明这些任务的执行器工作正常；记录在 `results/evaluator_positive_control.json`。这里的“完成”指实验训练、生成和评估完成，不等于 PBPF 比对照更好。']
else: lines += ['', '当前完整修复结果尚未完成；以 `repair_status.json` 为准。']
lines += ['', '## 兼容修正与验证', '',
'1. 清理 ROS 注入的 PYTHONPATH；测试使用 umask 077。',
'2. 将固定随机种子的零权重测试样例改为确定性粒子，额外覆盖精确零偏移的 CDF 边界；算法未改。',
'3. 显式启用 HF/Transformers 离线模式，避免 tokenizer 的附加模型查询绕过 local_files_only。',
'4. 补齐 Bubblewrap 内的 /bin 与 /sbin 链接，并只读挂载模型缓存。',
'5. 修复脚本不再向 actor 传入 labels 来计算随后不会使用的内部 loss，避免额外全词表 FP32 CE 的显存峰值。原本的混合损失计算保持不变。小型 Qwen2 检查表明移除该参数前后的 logits 逐位相同，记录见 `results/memory_fix_equivalence.json`。',
'6. 本地封装补充 Torch 初始化种子；原脚本的 --seed 没有固定 projector 初始化 RNG。',
'', '代码差异见 `compatibility.patch`。原始失败日志保留在 `logs/pytest.log`，修正后的全套结果在 `logs/pytest_compat.log`；最终代码再次完整验收为 671 passed，日志在 `logs/pytest_final.log`。',
'', '## 没有复现的部分与边界', '',
'**不是上游指标的精确数值复现**：上游报告的数据候选数为 282/54/79，本次为 353/89/94。仓库未附原始准备缓存和训练权重；本次使用当前默认参数，且显式补齐随机初始化控制。不能将相近指标称为完全一致。',
'**正式 H200/Slurm S0–S4 未运行**：实际执行 formal doctor 返回 exit 2，缺少 PBPF_ICLR_SITE、完整 scientific factory、独立 evaluator authority、数据/模型快照和集群资源配置。上游文档明确说明完整生产 factory 未提供。没有伪造这些条件或将合成流程当作正式实验。检查记录见 `logs/formal_doctor.log`。',
'', '## 常用命令', '', '```bash','bash local/verify.sh','bash local/run_smoke.sh','bash local/run_prediction.sh','bash local/run_repair.sh','```','',
'最后一条命令默认沿用已有 projector 权重并评估；要从零训练，应先保留并移开旧权重。预测脚本会覆盖同名本机结果。',
'服务器 /data I/O 延迟较高，模型缓存和运行中的大 checkpoint 使用 SSD；预测描述临时解压至隔离环境 tmpfs。未改变数据内容。']
(local/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Report rendered')
