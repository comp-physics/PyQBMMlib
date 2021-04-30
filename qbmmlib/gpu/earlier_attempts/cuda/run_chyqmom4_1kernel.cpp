#include "hyqmom.hpp"

float chyqmom4_single_kernel(float moments[], int num_moments,
                float xout[], float yout[], float wout[]);

/* print a usage message */
void usage(int argc, char **argv) {
    printf("usage: %s num_moments batch_size\n", argv[0]);
    return;
}

/* Set up inital moment inputs */
void init_input_6(float* moments, int size) {
    // data obtained from running stats.raw_gaussian_moments_bivar
    float one_moment[6] = {1.0, 1.0, 1.0, 1.01, 1.0, 1.01};
    for (int i = 0; i< size; i++) {
        for (int j = 0; j < 6; j++) {
            moments[i + j*size] = one_moment[j];
        }
    }
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        return 1;
    }
    int num_moments = atoi(argv[1]);
    int batch_size = atoi(argv[2]);

    float *input_moments = new float[6*num_moments];
    float *x_out_cuda = new float[4*num_moments];
    float *y_out_cuda = new float[4*num_moments];
    float *w_out_cuda = new float[4*num_moments];
    init_input_6(input_moments, num_moments);

    float cuda_time = chyqmom4_single_kernel(input_moments, num_moments, x_out_cuda, y_out_cuda, w_out_cuda);

    // for (int i=0; i<num_moments*4; i++) {
    //     // fprintf(stderr, "w[%d] = %f \n", i, w_out_cuda[i]);
    //     // fprintf(stderr, "x[%d] = %f \n", i, x_out_cuda[i]);
    //     // fprintf(stderr, "y[%d] = %f \n", i, y_out_cuda[i]);
    // }

    for (int j = 0; j < num_moments; j++) {
        try {
            if(fabs(w_out_cuda[0*num_moments + j] - 0.25) > 1e-3){throw 1;};
            if(fabs(w_out_cuda[1*num_moments + j] - 0.25) > 1e-3){throw 1;};
            if(fabs(w_out_cuda[2*num_moments + j] - 0.25) > 1e-3){throw 1;};
            if(fabs(w_out_cuda[3*num_moments + j] - 0.25) > 1e-3){throw 1;};
        } catch (int e) {
            fprintf(stderr, "w[0] got %f, expected %f\n", w_out_cuda[0*num_moments + j], 0.25);
            fprintf(stderr, "w[1] got %f, expected %f\n", w_out_cuda[1*num_moments + j], 0.25);
            fprintf(stderr, "w[2] got %f, expected %f\n", w_out_cuda[2*num_moments + j], 0.25);
            fprintf(stderr, "w[3] got %f, expected %f\n", w_out_cuda[3*num_moments + j], 0.25);
        }
    }
    printf("[CUDA] Took %f ms \n", cuda_time);
}