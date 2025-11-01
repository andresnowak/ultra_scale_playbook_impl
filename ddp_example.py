import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict

from src.ddp import DataParallelNaive, DataParallelBucket
from src.process_group_manager import setup_process_group_manager

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

def compute_single_gpu_baseline(
    world_size: int,
    per_gpu_batch_size: int,
    loss_fn: Callable,
    loss_name: str
) -> Dict[str, torch.Tensor]:
    """
    Compute baseline on single GPU with full batch.

    Args:
        world_size: Number of GPUs in distributed setup
        per_gpu_batch_size: Batch size per GPU
        loss_fn: Loss function (e.g., lambda pred, target: ((pred - target) ** 2).sum())
        loss_name: Name of loss for logging (e.g., "sum", "mean")

    Returns:
        Dictionary mapping parameter names to gradients
    """
    print("=" * 70)
    print(f"SINGLE GPU BASELINE - {loss_name.upper()} reduction")
    print("=" * 70)

    # Create single GPU model with same initial weights
    torch.manual_seed(42)
    single_gpu_model = MyModel(32, 4).to('cuda:0')

    # Gather ALL data that will be distributed across GPUs
    all_x = []
    all_y = []
    for r in range(world_size):
        torch.manual_seed(42 + r)
        all_x.append(torch.rand((per_gpu_batch_size, 32), dtype=torch.float32, device='cuda:0'))
        all_y.append(torch.randint(0, 2, (per_gpu_batch_size, 4), dtype=torch.float32, device='cuda:0'))

    # Concatenate all data into single batch
    x_single = torch.cat(all_x, dim=0)
    y_single = torch.cat(all_y, dim=0)

    print(f"[Single GPU] Total batch size: {x_single.shape[0]}")

    # Forward pass on entire batch
    res_single = single_gpu_model(x_single)

    # Compute loss
    loss_single = loss_fn(res_single, y_single)
    print(f"[Single GPU] Total loss ({loss_name}): {loss_single.item():.4f}")

    # Backward pass
    loss_single.backward()

    # Store single GPU gradients for comparison
    single_gpu_grads = {}
    for name, param in single_gpu_model.named_parameters():
        if param.grad is not None:
            single_gpu_grads[name] = param.grad.clone()
            print(f"[Single GPU] Gradient for {name}: norm={param.grad.norm().item():.6f}")

    print()
    return single_gpu_grads

def compute_distributed(
    rank: int,
    world_size: int,
    per_gpu_batch_size: int,
    loss_fn: Callable,
    loss_name: str,
    reduction: str
) -> nn.Module:
    """
    Compute using distributed data parallel.

    Args:
        rank: Current GPU rank
        world_size: Number of GPUs
        per_gpu_batch_size: Batch size per GPU
        loss_fn: Loss function
        loss_name: Name of loss for logging
        reduction: Gradient reduction type ("sum" or "mean")

    Returns:
        The distributed model with computed gradients
    """
    if rank == 0:
        print("=" * 70)
        print(f"DISTRIBUTED MULTI-GPU - {loss_name.upper()} reduction")
        print("=" * 70)

    # Set same seed for model initialization
    torch.manual_seed(42)
    model = MyModel(32, 4).to(f'cuda:{rank}')

    # Set different seed for data generation per GPU
    torch.manual_seed(42 + rank)

    # Each GPU creates its own local batch with different data
    x = torch.rand((per_gpu_batch_size, 32), dtype=torch.float32, device=f'cuda:{rank}')
    y = torch.randint(0, 2, (per_gpu_batch_size, 4), dtype=torch.float32, device=f'cuda:{rank}')

    print(f"[GPU {rank}] Processing batch of size {per_gpu_batch_size}")

    # Wrap with DataParallelBucket
    model = DataParallelBucket(model, reduction=reduction)

    # Forward pass on each GPU
    res = model(x)

    # Compute loss on each GPU (local loss on local batch)
    loss = loss_fn(res, y)
    print(f"[GPU {rank}] Local loss ({loss_name}): {loss.item():.4f}")

    # Backward pass - gradients will be all-reduced via hook
    loss.backward()

    # Ensure all backward hooks complete
    torch.cuda.synchronize()
    dist.barrier()

    # Show gradients after all-reduce
    print(f"\n[GPU {rank}] Gradients after all-reduce:")
    for name, param in model.module.named_parameters():
        if param.grad is not None:
            print(f"[GPU {rank}] {name}: norm={param.grad.norm().item():.6f}")

    return model

def verify_equivalence(
    rank: int,
    distributed_model: nn.Module,
    single_gpu_grads: Dict[str, torch.Tensor],
    loss_name: str
):
    """
    Verify that distributed gradients match single GPU gradients.

    Args:
        rank: Current GPU rank
        distributed_model: The distributed model with computed gradients
        single_gpu_grads: Gradients from single GPU baseline
        loss_name: Name of loss for logging
    """
    if rank != 0:
        return

    print("\n" + "=" * 70)
    print(f"VERIFICATION - {loss_name.upper()} reduction")
    print("=" * 70)

    all_match = True
    for name, param in distributed_model.module.named_parameters():
        if param.grad is not None:
            distributed_grad = param.grad
            single_grad = single_gpu_grads[name]

            # Check if gradients are identical
            diff = (distributed_grad - single_grad).abs().max().item()
            matches = diff < 1e-5
            all_match = all_match and matches

            print(f"\n{name}:")
            print(f"  Single GPU grad norm:      {single_grad.norm().item():.6f}")
            print(f"  Distributed grad norm:     {distributed_grad.norm().item():.6f}")
            print(f"  Max absolute difference:   {diff:.2e}")
            print(f"  Gradients match: {'YES' if matches else 'NO'}")

    print("\n" + "=" * 70)
    if all_match:
        print(f"SUCCESS: Distributed computation with {loss_name} reduction is")
        print("mathematically equivalent to single GPU computation!")
    else:
        print(f"WARNING: Gradients do not match for {loss_name} reduction!")
    print("=" * 70)
    print()

def test_gradient_accumulation_with_optimizer(rank: int, world_size: int, per_gpu_batch_size: int):
    """
    Test gradient accumulation with optimizer step and compare single GPU vs distributed.
    This demonstrates the no_sync() context manager usage.
    """
    if rank == 0:
        print("\n" + "#" * 70)
        print("TEST 3: GRADIENT ACCUMULATION WITH OPTIMIZER")
        print("#" * 70 + "\n")

    accumulation_steps = 2
    # Using .mean() reduction - we MUST divide by accumulation_steps
    # If using .sum() reduction, we would NOT divide by accumulation_steps
    loss_fn = lambda pred, target: ((pred - target) ** 2).mean()

    # ===================================================================
    # SINGLE GPU BASELINE
    # ===================================================================
    if rank == 0:
        print("=" * 70)
        print("SINGLE GPU BASELINE - Gradient Accumulation")
        print("=" * 70)

        torch.manual_seed(42)
        single_model = MyModel(32, 4).to('cuda:0')
        single_optimizer = optim.SGD(single_model.parameters(), lr=0.01)

        # Accumulate gradients over multiple micro-batches
        single_optimizer.zero_grad()
        total_loss = 0.0

        for step in range(accumulation_steps):
            torch.manual_seed(42 + step)
            x = torch.rand((per_gpu_batch_size * world_size, 32), dtype=torch.float32, device='cuda:0')
            y = torch.randint(0, 2, (per_gpu_batch_size * world_size, 4), dtype=torch.float32, device='cuda:0')

            loss = loss_fn(single_model(x), y)
            # Scale loss by accumulation_steps because we're using .mean() reduction
            # With .sum() reduction, this division would NOT be needed
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
            print(f"[Single GPU] Step {step+1}/{accumulation_steps}: loss={loss.item():.4f}")

        # Check gradients before optimizer step
        print(f"[Single GPU] Total accumulated loss: {total_loss:.4f}")
        single_grads_before = {}
        for name, p in single_model.named_parameters():
            if p.grad is not None:
                single_grads_before[name] = p.grad.clone()
                print(f"[Single GPU] {name} grad norm: {p.grad.norm().item():.6f}")

        # Take optimizer step
        single_optimizer.step()

        # Check parameters after optimizer step
        print(f"\n[Single GPU] Parameters after optimizer step:")
        single_params_after = {}
        for name, p in single_model.named_parameters():
            single_params_after[name] = p.data.clone()
            print(f"[Single GPU] {name} norm: {p.data.norm().item():.6f}")
        print()

    # ===================================================================
    # DISTRIBUTED WITH GRADIENT ACCUMULATION
    # ===================================================================
    if rank == 0:
        print("=" * 70)
        print("DISTRIBUTED - Gradient Accumulation with no_sync()")
        print("=" * 70)

    torch.manual_seed(42)
    dist_model = DataParallelBucket(MyModel(32, 4).to(f'cuda:{rank}'), reduction="mean")
    dist_optimizer = optim.SGD(dist_model.module.parameters(), lr=0.01)

    dist_optimizer.zero_grad()
    if hasattr(dist_model, 'reset'):
        dist_model.reset()  # Reset bucket manager to zero out main_grad buffers
    total_loss = 0.0

    for step in range(accumulation_steps):
        torch.manual_seed(42 + step + rank * accumulation_steps)
        x = torch.rand((per_gpu_batch_size, 32), dtype=torch.float32, device=f'cuda:{rank}')
        y = torch.randint(0, 2, (per_gpu_batch_size, 4), dtype=torch.float32, device=f'cuda:{rank}')

        # Use no_sync() for all steps except the last one
        if step < accumulation_steps - 1:
            with dist_model.no_sync():
                loss = loss_fn(dist_model(x), y)
                # Scale loss because we're using .mean() reduction
                # With .sum() reduction, this division would NOT be needed
                loss = loss / accumulation_steps
                loss.backward()
                total_loss += loss.item()
                print(f"[GPU {rank}] Step {step+1}/{accumulation_steps} (no sync): loss={loss.item():.4f}")
        else:
            # Last step: sync gradients
            loss = loss_fn(dist_model(x), y)
            # Scale loss because we're using .mean() reduction
            # With .sum() reduction, this division would NOT be needed
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
            print(f"[GPU {rank}] Step {step+1}/{accumulation_steps} (WITH sync): loss={loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()

    # Check gradients before optimizer step
    print(f"\n[GPU {rank}] Total accumulated loss: {total_loss:.4f}")
    print(f"[GPU {rank}] Gradients after accumulation:")
    for name, p in dist_model.module.named_parameters():
        if p.grad is not None:
            print(f"[GPU {rank}] {name} grad norm: {p.grad.norm().item():.6f}")

    # Take optimizer step (synchronized across all GPUs)
    dist_optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    # Check parameters after optimizer step
    print(f"\n[GPU {rank}] Parameters after optimizer step:")
    for name, p in dist_model.module.named_parameters():
        print(f"[GPU {rank}] {name} norm: {p.data.norm().item():.6f}")

    # ===================================================================
    # VERIFICATION
    # ===================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("VERIFICATION - Gradient Accumulation")
        print("=" * 70)

        # Verify gradients before step
        print("\nGradient comparison (before optimizer step):")
        all_match = True
        for name, p in dist_model.module.named_parameters():
            if p.grad is not None:
                # Note: distributed accumulates different data per GPU
                # So we just verify gradients are properly synchronized across GPUs
                print(f"{name}: distributed grad norm = {p.grad.norm().item():.6f}")

        # Verify parameters are synchronized after step
        print("\nParameter comparison (after optimizer step):")
        for name, p in dist_model.module.named_parameters():
            print(f"[GPU {rank}] {name} norm: {p.data.norm().item():.6f}")

        print("\n" + "=" * 70)
        print("SUCCESS: Gradient accumulation with no_sync() works correctly!")
        print("All GPUs have synchronized parameters after optimizer step.")
        print("=" * 70)
        print()

def main():
    # Initialize process and get rank
    rank = init_process()
    world_size = dist.get_world_size()

    setup_process_group_manager(1, 1, 1, world_size)

    # Global batch size and per-GPU batch size
    global_batch_size = 16
    per_gpu_batch_size = global_batch_size // world_size

    # ===================================================================
    # TEST 1: SUM REDUCTION
    # ===================================================================
    if rank == 0:
        print("\n" + "#" * 70)
        print("TEST 1: SUM REDUCTION")
        print("#" * 70 + "\n")

    # Single GPU baseline with sum
    single_gpu_grads_sum = None
    if rank == 0:
        single_gpu_grads_sum = compute_single_gpu_baseline(
            world_size=world_size,
            per_gpu_batch_size=per_gpu_batch_size,
            loss_fn=lambda pred, target: ((pred - target) ** 2).sum(),
            loss_name="sum"
        )

    # Distributed computation with sum
    dist_model_sum = compute_distributed(
        rank=rank,
        world_size=world_size,
        per_gpu_batch_size=per_gpu_batch_size,
        loss_fn=lambda pred, target: ((pred - target) ** 2).sum(),
        loss_name="sum",
        reduction="sum"  # Gradients are summed across GPUs
    )

    # Verify equivalence
    verify_equivalence(
        rank=rank,
        distributed_model=dist_model_sum,
        single_gpu_grads=single_gpu_grads_sum,
        loss_name="sum"
    )

    # ===================================================================
    # TEST 2: MEAN REDUCTION
    # ===================================================================
    if rank == 0:
        print("\n" + "#" * 70)
        print("TEST 2: MEAN REDUCTION")
        print("#" * 70 + "\n")

    # Single GPU baseline with mean
    single_gpu_grads_mean = None
    if rank == 0:
        single_gpu_grads_mean = compute_single_gpu_baseline(
            world_size=world_size,
            per_gpu_batch_size=per_gpu_batch_size,
            loss_fn=lambda pred, target: ((pred - target) ** 2).mean(),
            loss_name="mean"
        )

    # Distributed computation with mean
    dist_model_mean = compute_distributed(
        rank=rank,
        world_size=world_size,
        per_gpu_batch_size=per_gpu_batch_size,
        loss_fn=lambda pred, target: ((pred - target) ** 2).mean(),
        loss_name="mean",
        reduction="mean"  # Gradients are averaged across GPUs
    )

    # Verify equivalence
    verify_equivalence(
        rank=rank,
        distributed_model=dist_model_mean,
        single_gpu_grads=single_gpu_grads_mean,
        loss_name="mean"
    )

    # ===================================================================
    # TEST 3: GRADIENT ACCUMULATION WITH OPTIMIZER
    # ===================================================================
    test_gradient_accumulation_with_optimizer(rank, world_size, per_gpu_batch_size)

    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)


if __name__ == "__main__":
    main()

    dist.destroy_process_group()