
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

float qmom_cuda(float moments[], int num_moments, float *result_weight);
float qmom_openmp(float moments[], int num_moments,  float *result_weight, int nthread);
float qmom_naive(float moments[], int num_moments, float* result_weight);
