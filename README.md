# Parallel programming

## Elementary operations

### Broadcast

Here the idea is we have a root node and this root node will broadcast a piece of memory (here a tensor) to all available processes (gpus here)

### Reduce

In reduce we want to combine the data on each process through a function $f()$, so in this case somebody has to run this function $f()$ (there is no free-flying magic process which runs it), so in the case of reduce we will have root node that will receive the data of all processes and run the reduce operation with also its own data and only the root node will have this information

### All Reduce

In all reduce now instead we can have our processes in a Tree or ring formation, in the ring structure each rank (now will call it this instead of process) has a "next" and "previous" neighbor. In the ring formation each rank will send its data to the next neighbor and will also recive the data of its previous neignor and the rank will reduce the information it gets from its previous neigbhor and then send this data again to the next neighbor and after $P-1$ steps ($P$ being the size of the world), every rank will have the complete reduction of the data of all ranks

### Gather
For gather we can say this is similar to broadcast, but the difference is now we say that each rank has a piece of data and we want to gather all of that data in this case in one rank (our root). So here we create only a container list of data (here of tensors) on the root rank and then we gather all of the data from the whole world in the root rank

### All Gather
And for all gather the difference is that now instead al ranks will have a container list and everyone will gather the information of all ranks

(Here in NCCL all gather will also behave in ring structure from what I understand)


## Citations
```bibtex
@misc{ultrascale_playbook,
      title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
      author={Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf},
      year={2025},
}
```