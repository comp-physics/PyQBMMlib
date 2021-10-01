#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

float qmom_cuda(float moments[], int num_moments,
                float xout[], float yout[], float wout[]) ;
float qmom_openmp(float moments[], int num_moments, int nthread, 
                    float xout[], float yout[], float wout[]);
float qmom_naive(float moments[], int num_moments,
                    float xout[], float yout[], float wout[]);
