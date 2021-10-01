#include <cmath>
#include <cstdio>
#include <cassert>

#include "cudaErr.hpp"

__global__ void hyqmom3(float moments[], float x[], float w[], const int size, const int stride);
__global__ void hyqmom2(float moments[], float x[], float w[], const int size, const int stride);

float chyqmom4(float moments[], const int size, float w[], float x[], float y[], const int batch_size);
float chyqmom9(float moments[], const int size, float w[], float x[], float y[], const int batch_size, const int device_id);