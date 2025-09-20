import torch
import torch.distributed as dist


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(5, dtype=torch.float32).cuda()

    print(f"Before broadcast on rank {dist.get_rank()}: {tensor}")
    dist.broadcast(tensor, src=0)
    print(f"After broadcast on rank {dist.get_rank()}: {tensor}")


def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()

    print(f"Before reduce on rank {dist.get_rank()}: {tensor}")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"After reduce on rank {dist.get_rank()}: {tensor}")


def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()

    print(f"Before all reduce on rank {dist.get_rank()}: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all reduce on rank {dist.get_rank()}: {tensor}")


def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()

    if dist.get_rank() == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()
            for _ in range(dist.get_world_size())
        ]
    else:
        gather_list = None

    print(f"Before gather on rank {dist.get_rank()}: {gather_list}")
    dist.gather(tensor, gather_list, dst=0)
    print(f"After gather on rank {dist.get_rank()}: {gather_list}")


def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()

    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())
    ]

    print(f"Before gather on rank {dist.get_rank()}: {gather_list}")
    dist.all_gather(gather_list, tensor)
    print(f"After gather on rank {dist.get_rank()}: {gather_list}")


init_process()
example_all_gather()

dist.destroy_process_group()
