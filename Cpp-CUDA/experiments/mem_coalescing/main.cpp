#include "main.hpp"

/* print a usage message */
void usage(int argc, char **argv) {
    printf("usage: %s num_moments\n", argv[0]);
    return;
}

int main(int argc, char *argv[]) {
    /* CUDA */
    float* x_out_cuda, *w_out_cuda;
    float* x_out_cuda_2, *w_out_cuda_2;
    float* x_out_1, *w_out_1;
    float* x_out_2, *w_out_2;
    float* moment_d, *moment_2_d;
    float* moment_1_h, *moment_2_h;

    if (argc != 2) {
        usage(argc, argv);
        return 1;
    }
    int num_moments = atoi(argv[1]);

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
    printf("[NAIVE] took %f ms \n", naive_time);
    printf("[COAL] took %f ms \n", coal_time);
}

