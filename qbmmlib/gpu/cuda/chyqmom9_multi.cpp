#include <thread>
#include <vector>
#include "hyqmom.hpp"

/* print a usage message */
void usage(int argc, char **argv) {
    printf("usage: %s num_moments num_GPU\n", argv[0]);
    return;
}

/* Set up inital moment inputs */
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

    if (argc != 3) {
        usage(argc, argv);
        return 1;
    }
    int num_moments = atoi(argv[1]);
    int num_GPU = atoi(argv[2]);

    float *input_moments = new float[10*num_moments];
    float *x_out_cuda = new float[9*num_moments];
    float *y_out_cuda = new float[9*num_moments];
    float *w_out_cuda = new float[9*num_moments];
    init_input_10(input_moments, num_moments);

    // a vector of threads
    std::vector<std::thread> threads;
    int input_per_thread = num_moments / num_GPU;


    for (unsigned int device_id = 0; device_id < num_GPU; device_id++) {
        std::thread one_thread(chyqmom9, input_moments[device_id * input_per_thread], input_per_thread, 
                                    w_out_cuda[device_id * input_per_thread], 
                                    x_out_cuda[device_id * input_per_thread], 
                                    y_out_cuda[device_id * input_per_thread], 1, device_id);
        threads.push_back(one_thread);
    }

    for (auto &thread: threads) {
        thread.join();
    }

    // for (int i=0; i<num_moments*9; i++) {
    //     fprintf(stderr, "w[%d] = %f \n", i, w_out_cuda[i]);
    //     fprintf(stderr, "x[%d] = %f \n", i, x_out_cuda[i]);
    //     fprintf(stderr, "y[%d] = %f \n", i, y_out_cuda[i]);
    // }


    // for (int j = 0; j < num_moments; j++) {
    //     try {
    //         if(fabs(w_out_cuda[0*num_moments + j] - 0.027791) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[1*num_moments + j] - 0.111124) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[2*num_moments + j] - 0.027791) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[3*num_moments + j] - 0.111124) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[4*num_moments + j] - 0.444342) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[5*num_moments + j] - 0.111124) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[6*num_moments + j] - 0.027791) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[7*num_moments + j] - 0.111124) > 1e-3){throw 1;};
    //         if(fabs(w_out_cuda[8*num_moments + j] - 0.027791) > 1e-3){throw 1;};
    //     } catch (int e) {
    //         fprintf(stderr, "w[0] got %f, expected %f\n", w_out_cuda[0*num_moments + j], 0.027791);
    //         fprintf(stderr, "w[1] got %f, expected %f\n", w_out_cuda[1*num_moments + j], 0.111124);
    //         fprintf(stderr, "w[2] got %f, expected %f\n", w_out_cuda[2*num_moments + j], 0.027791);
    //         fprintf(stderr, "w[3] got %f, expected %f\n", w_out_cuda[3*num_moments + j], 0.111124);
    //         fprintf(stderr, "w[4] got %f, expected %f\n", w_out_cuda[4*num_moments + j], 0.444342);
    //         fprintf(stderr, "w[5] got %f, expected %f\n", w_out_cuda[5*num_moments + j], 0.111124);
    //         fprintf(stderr, "w[6] got %f, expected %f\n", w_out_cuda[6*num_moments + j], 0.027791);
    //         fprintf(stderr, "w[7] got %f, expected %f\n", w_out_cuda[7*num_moments + j], 0.111124);
    //         fprintf(stderr, "w[8] got %f, expected %f\n", w_out_cuda[8*num_moments + j], 0.027791);
    //     }
    // }
    // printf("[CUDA] Took %f ms \n", cuda_time);
}