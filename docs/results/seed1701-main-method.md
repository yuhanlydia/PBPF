# PBPF seed 1701 主方法结果（12/12 已评估）

每个单元固定 benchmark、模型、seed 1701 和 500 个 primary 来源。分数为每来源一个新候选的十项测试全通过数；Δ 是逐来源配对差值，95% CI 使用 10,000 次配对 bootstrap。

| Benchmark | 模型 | no_update | eesd_full | Δ 百分点 | 95% CI 百分点 |
|---|---|---:|---:|---:|---:|
| RunBugRun | Qwen2.5 Coder 7B | 228/500 | 240/500 | +2.4 | [-2.0, +6.8] |
| RunBugRun | DeepSeek Coder 6.7B | 184/500 | 198/500 | +2.8 | [-1.8, +7.4] |
| RunBugRun | Gemma 3 4B | 142/500 | 132/500 | -2.0 | [-5.6, +1.4] |
| CodeARC | Qwen2.5 Coder 7B | 85/500 | 83/500 | -0.4 | [-2.8, +2.0] |
| CodeARC | DeepSeek Coder 6.7B | 75/500 | 102/500 | +5.4 | [+2.8, +8.0] |
| CodeARC | Gemma 3 4B | 59/500 | 56/500 | -0.6 | [-2.6, +1.4] |
| APPS Replay | Qwen2.5 Coder 7B | 57/500 | 57/500 | +0.0 | [-1.8, +1.8] |
| APPS Replay | DeepSeek Coder 6.7B | 23/500 | 17/500 | -1.2 | [-3.2, +0.8] |
| APPS Replay | Gemma 3 4B | 36/500 | 2/500 | -6.8 | [-9.0, -4.8] |
| CodeContests Replay | Qwen2.5 Coder 7B | 47/500 | 39/500 | -1.6 | [-3.2, +0.0] |
| CodeContests Replay | DeepSeek Coder 6.7B | 18/500 | 24/500 | +1.2 | [-0.2, +2.8] |
| CodeContests Replay | Gemma 3 4B | 31/500 | 4/500 | -5.4 | [-7.6, -3.2] |

这是主方法表；其余方法消融尚在运行，最终双表须等待 84 个方法臂全部有评估结果或按协议标为不可估计。

Baseline manifest SHA-256: `c6892d1c9c520a542aedc47735ef58f33a2794c60b1297c6a0e271d292bf1e68`

## 主方法评估报告 SHA-256

- `runbugrun/qwen25_7b/eesd_full`: `8bdac0523a996dcd2ed3a4ea121af7122075a602b4e0b49c61dc06e5511e3c74`
- `runbugrun/deepseek_6p7b/eesd_full`: `0a6a649fc1c7a32fdafdb7906f3b9058dea1a80b7d185e2ee4f6537adfc7980b`
- `runbugrun/gemma3_4b/eesd_full`: `f21bd7c31a994849b2f45702c6f410ec34359fdef1880ac8bd214513d3929325`
- `codearc/qwen25_7b/eesd_full`: `7980ac70365e2b8cc7139e3012ff68ce22cda565eede543cb03de529e3c9cf64`
- `codearc/deepseek_6p7b/eesd_full`: `76fcdfe6a81f7f645c5fbc966ee09419aff9171d2792bbdd345fa1a78011095e`
- `codearc/gemma3_4b/eesd_full`: `ea4c7e43222de352456e7536d86b9035d6f854621711f5d25680a9f7522c192f`
- `apps_replay/qwen25_7b/eesd_full`: `d46235f38321322b0b69ba05750351ea1ed7c00eedc6b34a0023534bb8141e86`
- `apps_replay/deepseek_6p7b/eesd_full`: `806a8e1bf36373f7c2f6f217fa76512a101f36567c3a4fd66c3116e5187de8b1`
- `apps_replay/gemma3_4b/eesd_full`: `dc6188a28f3818482daf1619a5548bcffc6f1c3434a3d9ee8cbe0857a05ff464`
- `codecontests_replay/qwen25_7b/eesd_full`: `1bf207055a24b2e95e3c4a5d5ef07c7ae1e5799d3c79b4ed83e94862711cc5bf`
- `codecontests_replay/deepseek_6p7b/eesd_full`: `89a894ad0676a2b4690106fa628cdf4295bd8ff3f7667ec3e8a93ec9e5a9bf96`
- `codecontests_replay/gemma3_4b/eesd_full`: `3ab62b3138bdb71f8c2ba5ff6d3ccaa0122be44bf5b3b10b452c5c89e9226b3a`
