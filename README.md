# Parallel programming

When using torchrun each gpu will live in its own python process, with its own memory.

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

### Scatter 
For Scatter, this operation is similar to broadcast, but the difference is that instead of sending a whole copy of the rank tensor to everybody, here instead we send a slice of this tensor (or from our scatter list better said, we send one tensor) to each of the other nodes.

### Reduce Scatter
The difference of ReduceScatter to all reduce is that each node doesn't receive a whole copy of the output tensor of the other rank. Here in reduce scatter we instead scatter first a slice of the tensor of each node to every node, and then each of the nodes will apply a reduce operation with its corresponding slice and the slices it received from the other nodes.

Example:
```
world_size=2, op=SUM


- Rank 0’s input: [ [–1, 5], [2, 2] ]

- Rank 1’s input: [ [ 3, –3], [4, 1] ]
```

Then rank 0 will scatter `[2, 2]` to rank 1 and rank 1 will scatter `[3, -3]` to rank 0. After this we will have

```
- Rank 0's output: [2, 2]
- Rank 1's output: [6, 3]
``` 

We can see this operation in this steps:

- Scatter phase:
	- Each rank $r$ starts with a buffer divided into $P$ equal slices $$X^{(r)} = [\,X^{(r)}_0,\;X^{(r)}_1,\;\dots,\;X^{(r)}_{P-1}]\,.$$
	- Rank $r$ sends slice $i$ ($X^{(r)}_i$) to rank $i$.

- Local reduction phase
	- After the scatter, rank $k$ has received one slice from every rank: $$\{\,X^{(0)}_k,\;X^{(1)}_k,\;\dots,\;X^{(P-1)}_k\}\,.$$
	- Rank $k$ applies the reduction operation (sum, max, min, etc.) across that set, e.g. $$Y_k \;=\; \bigoplus_{r=0}^{P-1} X^{(r)}_k\,.$$
	- The result $Y_k$ is the only data that rank $k$ ends up with.

### A common implement of All Reduce is Ring All Reduce
The idea of this method is that rather than having all devices communicate with each other, which can cause communication bottlenecks, Ring all reduce instead employs a two step process; Reduce scatter and then All gather

- For reduce scatter we have:
	- Each device will split its data into N chunks (N being the world size) and sends one chunk of it to its neighbor
	- As each device receives its chunk it reduces its corresponding chunk with the one received from its neighbor
	- This will be repeated $N - 1$ times, and in the end each node will have a complete reduces slice
- For All gather we have:
	- Now that everyone has a reduced slice, we now start to send the completed slices
	- For each node we start on the position where we have our complete reduced slice and send it to our neighbor and at the same time we receive the reduce slice from our other neighbor
	- Then we send to our right neighbor the slice we received from our left neighbor and we continue like this for $N - 1$ again

Here each gpu sends $\frac{K}{N}$ values per transfer, where $K$ is the total number of values in the array being summed across GPUS (*so $\frac{K}{N}$ is the size of our slice*). Therefore, the total amount of data transferred to and from each GPU is $2 \times (N - 1) \times \frac{K}{N}$. When $N$ (the number of GPUS) is large, the amount of data transferred to and from each GPU is approximately $2 \times K$, where $K$ is the total number of parameters

So an All Reduce can be broken down into a Reduce Scatter followed by an All gather. And the communication cost of these operations is half of All Reduce, so in this case the communication cost for each operation is $K$.

### Barrier
This is a simple operation to synchronize all nodes. A barrier is not lifted until all nodes have reached the barrier, and only then after this happens are the nodes allowed to continue

## Citations
```bibtex
@misc{ultrascale_playbook,
      title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
      author={Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf},
      year={2025},
}
```