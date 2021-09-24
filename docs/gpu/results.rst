Performance results so far 
==========================

Cuda Vs Pycuda 
++++++++++++++

.. image:: fig/comet.png
    :width: 600

A comparison between PyCuda and Cuda. PyCuda is the python packaged used to access 
the GPU on Python. Both are compared with their CPU counter part. 

Chyqmom4 Vs Chyqmom9 Vs Chyqmom27
+++++++++++++++++++++++++++++++++

... where did the plot go?


Multi-GPU
+++++++++
Performance results on Chyqmom4 (top) and Chyqmom9 (bottom) on multiple GPU.

.. image:: fig/chyqmom4_multi.png
    :width: 600

The subplot shows the ratio between 1GPU and 2GPU. As number of input becomes 
sufficiently large, we see a 2x speed up when using 2 GPUs 

.. image:: fig/chyqmom9_multi.png
    :width: 600

CPU benchmark is gathered with Numba's just-in-time compiled python code. x
The subplot shows the ratio between GPU codes and the CPU Benchmark.

