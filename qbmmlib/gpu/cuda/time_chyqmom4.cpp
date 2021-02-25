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
    fprintf(stderr, "usage: %s filename max_input stride omp_nthreads\n", argv[0]);
}

/* Set up inital moment inputs for chyqmom4*/
void init_input_6(float moments[], int size) {
    // data obtained from running stats.raw_gaussian_moments_bivar
    float one_moment[6] = {1.0, 1.0, 1.0, 1.01, 1.0, 1.01};
    for (int i = 0; i< size; i++) {
        for (int j = 0; j < 6; j++) {
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
    float stride = std::stof(argv[3]);
    int omp_n_threads = atoi(argv[4]);

    std::ofstream result_file;
    char line[100];
    memset(line, 0, sizeof(char) * 100);
    result_file.open(filename);
    result_file << "Input Size, omp (s), cuda (s)\n";
    
    for (float x_moments = 1; x_moments < N_max; x_moments*= stride) {

        int num_moments = (int) ceil(x_moments);
        printf("Running %d inputs \n", num_moments);
        //input 
        float *input_moments = new float[6*num_moments];
        float *x_out_cuda = new float[4*num_moments];
        float *x_out_omp = new float[4*num_moments];
        float *y_out_cuda = new float[4*num_moments];
        float *y_out_omp = new float[4*num_moments];
        float *w_out_cuda = new float[4*num_moments];
        float *w_out_omp = new float[4*num_moments];
        init_input_6(input_moments, num_moments);

        // output results in column major format
        float omp_time = chyqmom4_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);
        // output results in row major format
        float cuda_time = chyqmom4(input_moments, num_moments, w_out_cuda, x_out_cuda, y_out_cuda, 1);

        // verify resultscompare
        #pragma omp parallel for
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < num_moments; col++) {
                float err_w = abs(w_out_omp[col * 4 + row] - w_out_cuda[row * num_moments + col]);
                float err_x = abs(x_out_omp[col * 4 + row] - x_out_cuda[row * num_moments + col]);
                float err_y = abs(y_out_omp[col * 4 + row] - y_out_cuda[row * num_moments + col]);
                if (err_w > 1e-3) {
                    // continue;
                    int i = row * num_moments + col;
                    printf("Error at: cuda[%d].x: %f ",i, w_out_cuda[i]);
                    printf("omp[%d].x: %f ",i, w_out_omp[i]);
                    printf("Err: %f \n", err_w);
                }
            }
        }
        sprintf(line, "%d,%f,%f\n", num_moments, omp_time, cuda_time);
        result_file << line;
        memset(line, 0, sizeof(char) * 100);
        
        delete[] input_moments;
        delete[] x_out_cuda;
        delete[] x_out_omp;
        delete[] y_out_cuda;
        delete[] y_out_omp;
        delete[] w_out_omp;
        delete[] w_out_cuda;
    }
    result_file.close();
    return 0;
}