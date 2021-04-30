#include <cstring>
#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream>

#include <omp.h>

#include "hyqmom.hpp"
#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s filename num_input max_batch stride omp_nthreads\n", argv[0]);
}

/* Set up inital moment inputs for chyqmom9*/
void init_input_10(float* moments, int size) {
    // data obtained from running stats.raw_gaussian_moments_bivar
    float one_moment[10] = {1, 1, 1, 1.01,  
                        1, 1.01, 1.03, 1.03,
                        1.0603, 1.0603};
    for (int i = 0; i< size; i++) {
        for (int j = 0; j < 10; j++) {
            moments[i + j*size] = one_moment[j];
        }
    }
}
int main(int argc, char **argv) {

    if (argc != 5) {
        usage(argc, argv);
        return 1;
    }

    std::string filename = argv[1];
    int N_max = atoi(argv[2]);
    int max_batch = atoi(argv[3]);
    float stride = std::stof(argv[4]);

    std::ofstream result_file;
    char line[100];
    memset(line, 0, sizeof(char) * 100);
    result_file.open(filename);
    result_file << "Batch Size,  cuda (s) | on " << N_max << " inputs" << std::endl;
    
    for (float x_batch = 1; x_batch < max_batch; x_batch += stride) {

        int N_batch = (int) ceil(x_batch);
        printf("Running %d baches on %d inputs \n", N_batch, N_max);
        //input 
        float *input_moments = new float[10*N_max];
        float *x_out_cuda = new float[9*N_max];
        float *y_out_cuda = new float[9*N_max];
        float *w_out_cuda = new float[9*N_max];
        init_input_10(input_moments, N_max);

        // 3 trials. final result is the min of three
        float cuda_time_1 = chyqmom9(input_moments, N_max, w_out_cuda, x_out_cuda, y_out_cuda, N_batch);
        float cuda_time_2 = chyqmom9(input_moments, N_max, w_out_cuda, x_out_cuda, y_out_cuda, N_batch);
        float cuda_time_3 = chyqmom9(input_moments, N_max, w_out_cuda, x_out_cuda, y_out_cuda, N_batch);

        float cuda_time = std::min(cuda_time_1, std::min(cuda_time_2, cuda_time_3));

        sprintf(line, "%d,%f\n", N_batch, cuda_time);
        result_file << line;
        memset(line, 0, sizeof(char) * 100);
        
        delete[] input_moments;
        delete[] x_out_cuda;
        delete[] y_out_cuda;
        delete[] w_out_cuda;
    }
    result_file.close();
    return 0;
}