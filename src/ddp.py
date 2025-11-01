import os
import contextlib
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.hooks import RemovableHandle
from torch.autograd import Variable

import src.process_group_manager as pgm
from .bucket import BucketManager


class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting
    point to understand how data parallelism works. It implements a simple
    all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """

    def __init__(self, module: nn.Module, reduction: str = "mean"):
        super().__init__()

        self.module = module
        self.reduction = reduction
        self.require_backward_grad_sync = True  # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation

        self.register_backward_hook(self._allreduce_grads)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require
        gradients.
        """
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def _allreduce_grads(self, param: torch.Tensor) -> None:
        """
        Performs an all-reduce operation to synchronize gradients across
        multiple processes.
        """
        if param.grad is None:
            return None

        if self.require_backward_grad_sync:
            dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                group=pgm.process_group_manager.cp_dp_group,
            )

            # https://discuss.pytorch.org/t/is-average-the-correct-way-for-the-gradient-in-distributeddataparallel-with-multi-nodes/34260/21
            if self.reduction == "mean":
                param.grad.div_(dist.get_world_size())

        return None

    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization.
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing
        gradients in between.
        """

        # When we exit the with block, no_sync iscalled again and we do to the True line
        self.require_backward_grad_sync = False
        yield 
        self.require_backward_grad_sync = True


class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the
    communication overhead.
    """

    def __init__(
        self, module, bucket_cap_mb=25, grad_type=torch.float32, reduction: str = "mean"
    ):
        """
        Initialize the DataParallelBucket module.
        Args:
        module (nn.Module): The model to be parallelized.
        process_group: The process group for gradient synchronization,
        which can be either a data parallel group or a context parallel
        group.
        bucket_cap_mb (int, optional): The maximum size of each gradient
        synchronization bucket in megabytes. Defaults to 25 MB.
        grad_type (torch.dtype, optional): The data type of gradients,
        defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True
        # whether to synchronize gradients during backward pass. Set to False
        # when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4  # float32 gradient: 4 bytes

        bucket_size = (
            bucket_cap_mb * 1024 * 1024 // grad_size
        )  # number of gradients in one bucket (MB)

        self.bucket_manager = BucketManager(
            module.parameters(),
            pgm.process_group_manager.cp_dp_group,
            bucket_size,
            grad_type,
        )

        self.register_backward_hook()
    
        # whether the callback for wait gradient synchronization is set
        self._post_backward_callback_set = False

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def backward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, output_tensor_grad: torch.Tensor):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize
        gradients.
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed
        precision.
        2. After gradient accumulation, it flags parameters as ready for
        synchronization.
        The gradient accumulation functions are stored to prevent them from
        going out of scope.
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.
        register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """

        self.grad_accs = []

        # nn.Parameter has requires_grad=True, it’s still a leaf in the autograd graph, so its .grad_fn is always None.  Autograd only attaches a grad_fn to non-leaf tensors—i.e. results of operations.

        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param) # Here param_tmp is a non leaf tensor now and its grad_fn is ExpandBackward
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]  # (function, index)
                # next_functions[0] points back to the node that will actually 
                # accumulate the gradient into the original leaf param, and 
                # second [0] grabs the first FunctionNode (in this case AccumulateGrad)
    
                # → ExpandBackward → AccumulateGrad(param) → …, we always have 
                # implicity accumulate_grad (param.grad += incoming_grad, 
                # because we have to accumulate gradients if the param is being 
                # called in multiple paths of the graph)

                # NOTE: This hook will be called after pytorch has accumulated the gradient from all paths in the computational graph into param.grad
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))

                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, param: torch.nn.Parameter, bucket_manager: BucketManager):
        """
        Creates the hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """

            if param.requires_grad:
                assert param.grad is not None

                param.main_grad.add_(param.grad.data) # Accumulates the gradients
                param.grad = None

                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    
                    if not self._post_backward_callback_set:
                        # NOTE: This hook will only be called after all the backward fns in the whole computational graph are finished, this is when we wait for all buckets to finish
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True

                    # mark the parameter as ready for gradient synchronization.
                    bucket_manager.mark_param_as_ready(param)

        return param_hook
    
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies
        the synchronized gradients back to the parameters' grad attribute.

        This method is called after the backward pass and before the optimizer step.
        """

        self.bucket_manager.wait()
        self._post_backward_callback_set = False

        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype)  # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 
