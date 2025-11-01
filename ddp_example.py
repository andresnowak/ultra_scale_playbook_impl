import torch
import torch.distributed as dist
import torch.nn as nn

from src.ddp import DataParallelNaive

def init_process():
    # Connect already running processes into the world
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

class MyModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.f1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.f1(x)

def main():
    # Initialize process and get rank
    rank = init_process()
    world_size = dist.get_world_size()

    # Each GPU gets different data
    torch.manual_seed(42 + rank)  # Different seed per GPU

    # Global batch size and per-GPU batch size
    global_batch_size = 16
    per_gpu_batch_size = global_batch_size // world_size

    # Each GPU creates its own local batch
    x = torch.rand((per_gpu_batch_size, 32), dtype=torch.float32, device=f'cuda:{rank}')
    y = torch.randint(0, 2, (per_gpu_batch_size, 4), dtype=torch.float32, device=f'cuda:{rank}')

    print(f"[GPU {rank}] Processing batch of size {per_gpu_batch_size}")
    print(f"[GPU {rank}] Input shape: {x.shape}")

    # Create model and move to GPU
    model = MyModel(32, 4).to(f'cuda:{rank}')

    # Wrap with DataParallelNaive
    model = DataParallelNaive(model, world_size, per_gpu_batch_size, reduction="sum")

    # Forward pass on each GPU
    res = model(x)
    print(f"[GPU {rank}] Output shape: {res.shape}")

    # Compute loss on each GPU
    loss = ((res - y) ** 2).sum()
    print(f"[GPU {rank}] Local loss: {loss.item():.4f}")

    # Synchronize and report
    dist.barrier()
    if rank == 0:
        print(f"\n[Master] All {world_size} GPUs completed processing")

if __name__ == "__main__":
    main()