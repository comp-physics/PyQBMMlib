Multi-GPU Implementation
========================

For sufficiently large input size, the performance can be further increased by 
dividing the workload evenly among multiple GPUs. In our case, the multi-GPU 
problem is quite simple, as our application does not require inter-GPU
communication. 

