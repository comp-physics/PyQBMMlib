.. _gpu-start:

GPU Implementation of CHYQMOMs 
==============================

The `qbmmlib/gpu` folder contains code that offload the CHYQMOMs to Nvidia's 
GPU in hope that GPU can significantly speed up these algorithms due to their
SIMD nature. 

The earliest experiments are implemented in Cuda, and were compared
with their C counterparts. These experiments are contained in
`qbmmlib/gpu/cuda` folder. The code themselves are built with `cmake`

Later, the GPU code are implemented in python with the package `pyCuda`


.. toctree::
   :maxdepth: 2

   gpu-mem.rst
   gpu-spec.rst
   multi-gpu.rst

