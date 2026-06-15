# peft_learning

参数高效微调（PEFT）学习，以 BERT 文本分类为任务，实践 LoRA 等方法，并对比单卡与 Accelerate 训练。

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `peft_util.py` | 数据加载（`get_loader`）与模型构建（`get_model`）工具 |
| `peft.single.py` | 单卡训练 / 加载 PEFT 模型示例 |
| `peft.accelerate.py` | Accelerate（可叠加 DeepSpeed）下的 LoRA 微调 |
| `tokenizer/` | 分词器文件 |

## 依赖

`torch`、`peft`、`modelscope`、`accelerate`。

## 运行

```bash
python peft.single.py                 # 单卡
accelerate launch peft.accelerate.py  # 多卡 / DeepSpeed
```

## 学习要点

- `LoraConfig` / `get_peft_model` 的用法与可训练参数占比
- `save_pretrained` / `PeftModel.from_pretrained` 的适配器保存与加载
- LoRA 与全量微调在显存和效果上的差异
