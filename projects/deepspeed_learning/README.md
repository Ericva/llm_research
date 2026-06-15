# deepspeed_learning

对照学习三种分布式训练范式，模型统一为同一个 `SimpleNet` MLP，便于横向比较 API 与配置差异。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `deepspeed.1.py` | 原生 DeepSpeed API（`deepspeed.initialize`）训练示例 |
| `ds.config` | DeepSpeed 配置（ZeRO stage、optimizer、batch 等） |
| `fsdp.1.py` | PyTorch 原生 FSDP（`FullyShardedDataParallel`）+ 分布式 state_dict 保存 |
| `accelerate.1.py` | HuggingFace Accelerate 封装 DeepSpeed 的训练示例 |
| `run.deepspeed.sh` | DeepSpeed 启动脚本 |
| `run.fsdp.sh` | FSDP（torchrun）启动脚本 |
| `run.accelerate.sh` | Accelerate 启动脚本 |

## 运行

```bash
bash run.deepspeed.sh    # 原生 DeepSpeed
bash run.fsdp.sh         # PyTorch FSDP
bash run.accelerate.sh   # Accelerate + DeepSpeed
```

## 学习要点

- ZeRO 三个 stage 对显存 / 通信的取舍
- FSDP 的 sharding 策略与 state_dict 收集（Full vs Local）
- Accelerate 如何在不改训练循环的前提下接入 DeepSpeed
