GPU Memory
==========
Cuda GPUs have a number of different memory components available. The ones that 
concern us are: 

- registers: used by each thread to store local variables.
- shared memory: memory shared among threads in a single block, cannot be accessed 
  by threads in other blocks
- global memory: GPU device memory
- host memory: CPU memory 

One important factor that determines the GPU performance is the GPU's memory 
access pattern. When a GPU thread request data from the global memory, the GPU 
will initiate a *transaction* between the thread and the global memory. The 
size of the transaction is always the same, no matter how much data the thread 
actually requested. However, when multiple threads request data that are consecutive 
in memory, the data can be delivered in a single transaction. To achieve highest 
performance, it is important to allocate memory in a way such that GPU threads 
can request consecutive memory at the same time 

Originally, the input moment 2x2 arrays are formatted such that each row is a single moment, 
and number of rows are number of moments. When stored in memory as continuos array, each
moment are consecutive to one another. During execution, each GPU thread will request the first 
element of each moment. However, since the size of each memory transactions are fixed, the GPU 
must also include rest of the moment elements in the transaction, which the threads do not 
need at this time. This results in a wasted data transfer, and will increase the number of transaction
required. 

In the GPU implementations, input moments are formatted such that each column is a single moment, 
and the number of columns are the number of moments. In memory, this format will ensure that the first 
element of all moments are consecutive to one another. During execution, each GPU thread will request 
the first element of each moment, and since all first elements are consecutive in memory, the GPU can 
contain them in a single transaction. This results in better performance, as the number of total 
transactions is reduced. 
