# 本次完整本地修复诊断

在固定 Qwen2.5-Coder-7B-Instruct 权重校验、2 步集成检查通过后运行：

```bash
bash local/run_repair.sh
```

- 模型版本：c03e6d358207e414f1eca0bb1891e29f1db0e242，4-bit。
- belief：本次当前默认数据上训练的 seed 1703 权重。
- projector：从头训练 1500 步；8 prefix tokens；learning rate 3e-5；sequence cap 1024；prefix RMS 0.02、delta RMS 0.002。
- 每 250 步评估 development NLL，按 development 选择权重。
- 8 个 test 候选程序，6 个条件，greedy，最多生成 512 tokens。
- `local/run_repair_seeded.py` 设置 Python/NumPy/Torch 种子为 2701；原始脚本的 seed 参数没有固定 projector 初始化用的 Torch RNG。本地封装补充这一项，不改科学算法。
- 不会以 smoke 权重作为完整训练的起点；两者分别保存。
- 数据划分仍是本次默认准备得到的 353/89/94，与上游 282/54/79 不同。
- 结果为本机完整小规模修复诊断，不是上游指标的精确匹配，也不是正式 H200 S0–S4 验证。
- 正负结果均保留，不依据 test 结果反复调参。
