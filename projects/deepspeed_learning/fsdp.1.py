import os, torch, torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, \
    LocalStateDictConfig


class SimpleNet(torch.nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hid)
        self.fc2 = torch.nn.Linear(d_hid, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# def setup_dist():
#    dist.init_process_group("nccl", init_method="env://")
#    local_rank = int(os.environ["LOCAL_RANK"])
#    torch.cuda.set_device(local_rank)
#    return local_rank, dist.get_rank(), dist.get_world_size()


def setup_dist():
    # torchrun 会注入这几个环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1) 固定本进程使用的 GPU
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # 2) 绑定 device 到进程组（关键！消除告警）
    #    注：device_id 在 torch>=2.6 可用
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,  # 绑定 PG 的设备，避免 barrier 警告
    )

    # 3) 如果你代码里会用 barrier，显式传 device_ids 更保险（可选）
    dist.barrier(device_ids=[local_rank])

    return local_rank, dist.get_rank(), dist.get_world_size(), device


def save_full_state_dict(model: FSDP, path: str):
    cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(sd, path)


def save_local_state_dict(model: FSDP, dirpath: str):
    cfg = LocalStateDictConfig()
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, cfg):
        sd = model.state_dict()
    os.makedirs(dirpath, exist_ok=True)
    torch.save(sd, os.path.join(dirpath, f"shard_rank{dist.get_rank()}.pt"))


def main():
    local_rank, rank, world_size, device = setup_dist()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)

    input_dim = 10
    hidden_dim = 256
    output_dim = 2
    batch_size = 64
    data_size = 10000
    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)

    data_set = TensorDataset(input_data, labels)
    sampler = DistributedSampler(data_set, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(data_set, batch_size=batch_size, sampler=sampler)

    model = SimpleNet(input_dim, hidden_dim, output_dim)
    model = FSDP(model.to(device))
    optimizator = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(500):
        model.train()
        sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizator.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizator.step()

        if epoch % 50 == 0 and rank == 0:
            print(f"epoch {epoch} loss {loss.item():.6f}")

    if rank == 0:
        os.makedirs("fsdp_ckpts", exist_ok=True)
    dist.barrier()

    save_full_state_dict(model, "fsdp_ckpts/full_state_dict.pt")  # A) 聚合后单文件
    save_local_state_dict(model, "fsdp_ckpts/local_shards")  # B) 每 rank 一片
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()