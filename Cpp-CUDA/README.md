CHYQMOM4 Parallelization Experiments 
====================================

This folder contains code for chyqmom4, implemented in CUDA and C++ with OpenMP

Compile the code
----------------

I use CMake to build my program. The minimum version requirement is 
arbitrary, set at 3.16. if you are on Ubuntu 20, you should be able to 
install this version with: 

    sudo apt-get install cmake

Run the following command: 

    # make a directory caled "build" and build the project in it
    # make sure the name is "build" so git does not track it
    mkdir build 
    cd build 
    cmake ..
    make 

Execute the code 
----------------

By now there should be an executable called `test_qmom` in the `build` folder.
It takes 2 input arguments: first the number of moments to process (input size), 
second the number of threads to run for OpenMP. Run an example with 1000 sets of
input moments with 10 OpenMP threads:

    ./test_qmom 1000 10

The input size can be as large as you want, until your GPU runs out of memory. 
For me this happens at around 25 000 000. I also find that setting the input 
size as a multiple of thread number gives the best performance for OpenMP.

What the code is doing
----------------------
The code will attempt to run chyqmom4 on a array of input moments of the specified size
in three ways: with GPU, with multithreaded OpenMP, and with single threaded loop. 
The main entry point of the code is `main.cpp`. All GPU implementation is in `qmom_cuda.cu`, 
which is just a C++ rewrite of the `qbmm_pyuda.py` file.
All CPU implementation (OpenMP and Naive) is in `qmom_openmp.cpp`, which is just a slight modification 
of the C code Spencer wrote. `cudaErr.hpp` file contains a helpful macro that prints any encountered 
GPU error (otherwise there is no indication of the error if it occurs)



