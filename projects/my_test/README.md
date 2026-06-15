# my_test

并行原语手写实验场，验证对 TP（张量并行）/ DP（数据并行）的理解。

## ray_tp_dp_test/

基于 [Ray](https://docs.ray.io/) 的 placement group + scheduling strategy，手动编排并行训练。

| 文件 | 说明 |
| --- | --- |
| `ray_tp.py` | 纯张量并行（TP）实验 |
| `ray_tp_dp.py` | TP 与 DP 组合（2D 并行）实验 |
| `ray_tp.sh` | 启动脚本 |

## 运行

```bash
bash ray_tp_dp_test/ray_tp.sh
```

## 学习要点

- 用 Ray placement group 将 actor 绑定到指定 GPU
- 手动设置 `RANK` / `WORLD_SIZE` 并初始化 `torch.distributed`
- TP 内部按列 / 行切分权重，DP 按 batch 切分数据
