#include <cassert>
#include "main.hpp"


/* print a usage message */
void usage(int argc, char **argv) {
    printf("usage: %s num_moments num_thread \n", argv[0]);
    return;
}

/* Set up inital moment inputs */
void init_input(float* moments, int size) {

    float one_moment[] = {1, 1, 1, 1.01, 1, 1.01};
    for (int i = 0; i< size * 6; i+= 6) {
        memcpy((void*)&moments[i], &one_moment, sizeof(float) * 6);
    }
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        return 1;
    }
    int num_moments = atoi(argv[1]);
    int num_thread = atoi(argv[2]);

    float *input_moments = new float[6*num_moments];
    float *result_weight_cuda = new float[4*num_moments];
    float *result_weight_omp = new float[4*num_moments];
    float *result_weight_naive = new float[4*num_moments];
    init_input(input_moments, num_moments);

    float cuda_time = qmom_cuda(input_moments, num_moments, result_weight_cuda);
    float omp_time = qmom_openmp(input_moments, num_moments, result_weight_omp, num_thread);
    float naive_time = qmom_naive(input_moments, num_moments, result_weight_naive);

    // verify results
    for (int i = 0; i < num_moments*4; i++) {
        assert(result_weight_cuda[i] == result_weight_omp[i]);
    }

    printf("[CUDA]    Took %f ms \n", cuda_time);
    printf("[OPEN_MP] Took %f ms \n", omp_time);
    printf("[NAIVE]   Took %f ms \n", naive_time);
}