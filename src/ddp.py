import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.hooks import RemovableHandle

class DataParallelNaive(nn.Module):
    """
    We will do first forward, then backward and then the all reduce (no optimizer)
    """

    def __init__(self, module: nn.Module, world_size: int, mbs: int, reduction: str="mean"):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.reduction

        self.register_backward_hook(self._allreduce_grads)

        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def register_backward_hook(self, hook):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

          
    def _allreduce_grads(self, grad: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        if self.reduction == "mean":
            grad /= self.world_size

        return grad