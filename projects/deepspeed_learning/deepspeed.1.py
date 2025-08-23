import deepspeed
import torch
from torch.utils.data import TensorDataset

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='deepspeed training script.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)   # 注入 --deepspeed / --deepspeed_config
    return parser.parse_args()

if __name__ == '__main__':
    # ---- toy data ----
    input_dim, hidden_dim, output_dim = 10, 256, 2
    data_size = 10_000
    #batch_size = 64
    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)        # MSE → 浮点标签
    dataset = TensorDataset(input_data, labels)

    #dataloader = DataLoader(data_set, batch_size = batch_size)
    # ---- model ----
    model = SimpleNet(input_dim, hidden_dim, output_dim)

    # ---- args & distributed ----
    args = parse_arguments()
    deepspeed.init_distributed()

    # ---- DeepSpeed 初始化：让它代建 DataLoader（batch 来自 ds.json 的 micro）----
    engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset         # ← 传 Dataset，不要自己建 DataLoader
        # 不传 optimizer → DeepSpeed 按 ds.json 创建
    )

    criterion = torch.nn.MSELoss()
    model_dtype = next(engine.module.parameters()).dtype   # bf16 / float32

    for epoch in range(100):
        engine.train()
        for inputs, labels in train_loader:
            xb = inputs.to(engine.device, dtype=model_dtype, non_blocking=True)
            yb = labels.to(engine.device, dtype=model_dtype, non_blocking=True)  # MSE → 浮点

            outputs = engine(xb)
            loss = criterion(outputs, yb)
            engine.backward(loss)
            engine.step()

        if (epoch + 1) % 10 == 0 and engine.global_rank == 0:
            print(f"[epoch {epoch + 1}] loss={loss.item():.6f}")

        if (epoch + 1) % 50 == 0:  # 所有 rank 都要调用
            engine.save_checkpoint("ckpts", tag=f"ep{epoch + 1}")

    # 优雅关闭
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()