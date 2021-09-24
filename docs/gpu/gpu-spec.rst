Relevant GPU Specifications 
===========================

+------------------------+--------------------+------------+------------+-------------+
| Specifications         | Pascal GTX 1050 Ti | Turing T4  | Kepler K80 | Pascal P100 |
+========================+====================+============+============+=============+
| Launch Date            | 01-03-2017         | 09-12-2018 | 11-17-2014 | 06-20-2016  |
+------------------------+--------------------+------------+------------+-------------+
| Compute Capability     | 6.1                | 7.5        | 3.7        | 6.0         |
+------------------------+--------------------+------------+------------+-------------+
| Threads per block      | 1024               | 1024       |            | 1024        | 
+------------------------+--------------------+------------+------------+-------------+
| Threads per MP         | 2048               | 1024       |            | 2048        | 
+------------------------+--------------------+------------+------------+-------------+
| Register per block     | 65536              | 65536      |            | 65536       | 
+------------------------+--------------------+------------+------------+-------------+
| Register per MP        | 65536              | 65536      |            | 65536       | 
+------------------------+--------------------+------------+------------+-------------+
| 32bit FLOP/s           | 1.982 T            | 8.141 T    | 8.73 T     |             |
+------------------------+--------------------+------------+------------+-------------+
| 64bit FLOP/s           | 61.94 G            | 254.4 G    | 2.91 T     |             | 
+------------------------+--------------------+------------+------------+-------------+
| Multiprocessors (MC)   | 6                  | 40         |            | 56          | 
+------------------------+--------------------+------------+------------+-------------+
| Clock rate             | 1.29 GHz           | 1.59GHz    |            |             |
+------------------------+--------------------+------------+------------+-------------+
| Concurrent Kernels     | Yes                | Yes        |            |             |
+------------------------+--------------------+------------+------------+-------------+
| Thread per warp        | 32                 | 32         |            |             |
+------------------------+--------------------+------------+------------+-------------+
| Global memory bandwitdh| 112.128 GB/s       | 300 GB     |            |             | 
+------------------------+--------------------+------------+------------+-------------+
| Global memory size     | 3.938 GB           | 16 GB      |            |             |
+------------------------+--------------------+------------+------------+-------------+
| Memcpy Engines         | 2                  | 3          |            |             | 
+------------------------+--------------------+------------+------------+-------------+

Notes on Attributes: 
  - Compute capability is a important parameter for compiling the CUDA code. GPU will 
    not be able to run code compiled with a different compute capability 
  - Registers per multiprocessors may become a limiting factor on performance if a single 
    thread takes up too many registers (local variables)
  - It is important to know whether concurrent kernels is supported by the GPU, for this 
    will impact the optimization process
  - Memcpy engine determines how much memcpy/kernel overlap can be achieved 

Performance Comparison on different hardware:
+++++++++++++++++++++++++++++++++++++++++++++ 

.. image:: fig/chyqmom9_k80_comet.png
  :width: 600

.. image:: fig/chyqmom9_p100_comet.png
  :width: 600

Chyqmom9 Cuda performance on Kepler K80 (top) and Pascal P100 (bottom). 
Data is gather from Xsede Comet, and the CPU benchmark 
is implemented with OpenMP on 24 cores. 
For each plot, the bottom subplot is OpenMP time : Cuda time ratio. 
