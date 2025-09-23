import torch
import torch.distributed as dist
import time


def init_process():
    # here we connect the already running processes into the world
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


def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
        ]  # so when we say slices, we say that each gpu will get one of the tensors on the scatter list
    else:
        scatter_list = None

    tensor = torch.zeros(5, dtype=torch.float32).cuda()

    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")


def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()
        ** (
            j + 1
        )  # Offset based on node rank and then we do a power exponent based on the number of each rank in the world
        for j in range(world_size)
    ]

    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After ReduceScatter on rank {rank}: {output_tensor}")

def example_barrier():
    rank = dist.get_rank()
    t_start = time.time()
    print(f"Rank {rank} sleeps for {rank} seconds")
    time.sleep(rank) # simulate a work load
    dist.barrier(device_ids=[rank]) # Here we will wait until all the nodes call the barrier
    print(f"Rank {rank} after barrier time delta: {time.time() - t_start:.4f}")

def example_ring_all_reduce():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunks = [
        torch.tensor([rank + 1], dtype=torch.float32).cuda()
        for _ in range(world_size)
    ]

    # neighbors in the ring
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    print(f"Before all reduce on rank {dist.get_rank()}: {torch.concat(chunks, dim=-1)}")

    # 1) Reduce-Scatter
    # Each iteration k we reduce chunk (rank - k) into our local buffer.
    for k in range(world_size - 1):
        # -1 % 3 = 2, -2 % 3 = 1, -3 % 3 = 0
        send_idx = (rank - k) % world_size
        recv_idx = (rank - k - 1) % world_size

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(send_buf)

        # kick off non-blocking send/recv
        req_send = dist.P2POp(dist.isend, send_buf, peer=right)
        req_recv = dist.P2POp(dist.irecv, recv_buf, peer=left)
        # We wait until our two neighbors do what we asked them to do
        reqs = dist.batch_isend_irecv([req_send, req_recv])
        for req in reqs:
            req.wait()

        chunks[recv_idx] += recv_buf

    # After the (world_size - 1) operations each node will have one the slices completely reduced

    # 2) All gather
    for k in range(world_size - 1):
        send_idx = (rank - k + 1) % world_size
        recv_idx = (rank - k) % world_size

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(send_buf)

        req_send = dist.P2POp(dist.isend, send_buf, peer=right)
        req_recv = dist.P2POp(dist.irecv, recv_buf, peer=left)
        # wait both
        reqs = dist.batch_isend_irecv([req_send, req_recv])
        for req in reqs:
            req.wait()

        chunks[recv_idx] = recv_buf

    print(f"After all reduce on rank {dist.get_rank()}: {torch.concat(chunks, dim=-1)}")


init_process()
example_ring_all_reduce()

dist.destroy_process_group()
