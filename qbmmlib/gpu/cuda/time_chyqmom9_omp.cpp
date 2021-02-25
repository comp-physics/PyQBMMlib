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
    float stride = std::stof(argv[3]);
    int omp_n_threads = atoi(argv[4]);

    std::ofstream result_file;
    char line[100];
    memset(line, 0, sizeof(char) * 100);
    result_file.open(filename);
    result_file << "Input Size, OMP1 (ms), OMP2 (ms), OMP3 (ms), OMP4 (ms), OMP5 (ms)\n";
    
    for (float x_moments = 1; x_moments < N_max; x_moments*= stride) {

        int num_moments = (int) ceil(x_moments);
        printf("Running %d inputs \n", num_moments);
        //input 
        float *input_moments = new float[10*num_moments];
        float *x_out_omp = new float[9*num_moments];
        float *y_out_omp = new float[9*num_moments];
        float *w_out_omp = new float[9*num_moments];
        init_input_10(input_moments, num_moments);

        // output results in column major format
        float omp_time1 = chyqmom9_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);
        float omp_time2 = chyqmom9_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);
        float omp_time3 = chyqmom9_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);
        float omp_time4 = chyqmom9_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);
        float omp_time5 = chyqmom9_omp(input_moments, num_moments, omp_n_threads, x_out_omp, y_out_omp, w_out_omp);

        sprintf(line, "%d, %f, %f, %f, %f, %f\n", num_moments, omp_time1, omp_time2, omp_time3, omp_time4, omp_time5);
        result_file << line;
        memset(line, 0, sizeof(char) * 100);
        
        delete[] input_moments;
        delete[] x_out_omp;
        delete[] y_out_omp;
        delete[] w_out_omp;
    }
    result_file.close();
    return 0;
}