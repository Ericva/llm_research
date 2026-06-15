# llm_research

LLM / 分布式训练 / 推理 / Agent 方向的**个人学习工作区**。

本仓库分为两类内容：

- **`external/`** —— 从他人远端拉取的第三方代码库（submodule 或独立 clone），用于阅读、跑通、对照学习。**不在此处改动其内容**，升级走各自的上游。
- **`projects/`** —— 我自己编写的学习代码，每个子目录是一个独立的学习主题，多为「跑通 + 注释 + 复现」性质。

---

## 目录总览

### external/ — 第三方学习库

| 目录 | 来源 | 说明 | 纳管方式 |
| --- | --- | --- | --- |
| `nano-vllm` | [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 极简版 vLLM 实现，学习推理引擎核心 | git submodule |
| `spring-2025-lectures` | Stanford CS336 | 课程讲义与示例代码 | 本地 clone（`.gitignore` 忽略） |
| `learn-harness-engineering` | walkinglabs 课程 | AI coding agent 的 Harness 工程实践课 | 独立 git 仓库（含自有 `.git`，未纳入父仓库） |

### projects/ — 自有学习项目

| 目录 | 主题 | 说明 |
| --- | --- | --- |
| [`learn_hermes`](projects/learn_hermes/README.md) | Agent 系统设计 | 仿照 hermes-agent 从零搭建的精简版 Mini Agent（里程碑制） |
| [`cs336_learning_2026`](projects/cs336_learning_2026/README.md) | LLM from scratch | 跟随 Stanford CS336 的学习记录 |
| [`deepspeed_learning`](projects/deepspeed_learning/README.md) | 分布式训练 | DeepSpeed / FSDP / Accelerate 三种并行训练范式对照 |
| [`peft_learning`](projects/peft_learning/README.md) | 参数高效微调 | LoRA 等 PEFT 方法 + 单卡 / Accelerate 训练 |
| [`vllm_learning`](projects/vllm_learning) | 推理服务 | vLLM server / client 使用（git submodule） |
| [`my_test`](projects/my_test/README.md) | 并行原语实验 | 基于 Ray 的 TP / DP 手写实验 |
| [`env_setting`](projects/env_setting/README.md) | 环境配置 | conda / pip / ssh / zsh 一键配置脚本 |

---

## Submodule 使用

首次 clone 后拉取 submodule：

```bash
git submodule update --init --recursive
```

当前 submodule（见 `.gitmodules`）：

- `external/nano-vllm`
- `projects/vllm_learning`

---

## 约定

- 第三方库只读不改；要做实验请在 `projects/` 下新建主题目录。
- 每个自有项目目录都应有自己的 `README.md`，说明**学习目标、文件清单、运行方式**。
- 敏感信息（`.env`、密钥）已在 `.gitignore` 中忽略，不要提交。
