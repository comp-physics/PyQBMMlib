#include <cstring>
#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream>

#include "main.hpp"

/* print out a usage message */
void usage(int argc, char **argv) {
    fprintf(stderr, "usage: %s filename max_size stride\n", argv[0]);
}

int main(int argc, char **argv) {

    if (argc != 4) {
        usage(argc, argv);
        return 1;
    }

    std::string filename = argv[1];
    int N_max = atoi(argv[2]);
    int stride = atoi(argv[3]);

    std::ofstream result_file;
    char line[100];
    memset(line, 0, sizeof(char) * 100);
    result_file.open(filename);
    result_file << "Input Size, Naive, Coalesced \n";

    for (int size=1; size< N_max; size+=stride) {
        printf("Calculating for size = %d \n", size);
        float* x_out_cuda, *w_out_cuda;
        float* x_out_cuda_2, *w_out_cuda_2;
        float* x_out_1, *w_out_1;
        float* x_out_2, *w_out_2;
        float* moment_d, *moment_2_d;
        float* moment_1_h, *moment_2_h;

        int num_moments = size;

        x_out_1 = new float[num_moments*2];
        x_out_2 = new float[num_moments*2];
        w_out_1 = new float[num_moments*2];
        w_out_2 = new float[num_moments*2];
        moment_2_h = new float[num_moments*3];
        moment_1_h = new float[num_moments*3];
        init_input_row_major(moment_1_h, num_moments);
        init_input_col_major(moment_2_h, num_moments);
    
        float naive_time = run_naive(moment_1_h, num_moments, x_out_1, w_out_1);
        float coal_time = run_coal(moment_2_h, num_moments, x_out_2, w_out_2);

        for (int row = 0; row < 2; row++) {
            for (int col=0; col < num_moments; col++) {
                assert(x_out_2[row * num_moments + col] == x_out_1[2*col + row]);
            }
        }
        sprintf(line, "%d,%f,%f\n", size, naive_time, coal_time);
        result_file << line;
        memset(line, 0, sizeof(char) * 100);

        delete[] x_out_1;
        delete[] x_out_2;
        delete[] w_out_1;
        delete[] w_out_2;
        delete[] moment_1_h;
        delete[] moment_2_h;
    }
    result_file.close();
    return 0;
}

