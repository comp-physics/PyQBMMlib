#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>
#include <stdbool.h>

float hyqmom2_cuda(float moments[], int num_moments);

void hyqmom2(float mom[], float x[], float w[]) {
    float bx, d2, c2;

    w[0] = mom[0] / 2.0;
    w[1] = w[0];

    bx = mom[1] / mom[0];
    d2 = mom[2] / mom[0];
    c2 = d2 - pow(bx,2.0);

    x[0] = bx - sqrt(c2);
    x[1] = bx + sqrt(c2);
}

float hyqmom2_openmp(float moments[], int num_moments, int nthread)
{
    omp_set_num_threads(nthread);
    float *xout = new float[num_moments*2]; 
    float *wout = new float[num_moments*2]; 

    printf("[OPEN_MP] starting %d thread(s) \n", omp_get_max_threads());
    printf("[OPEN_MP] starting loop. Timer on... \n");

    double tic = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0; i<num_moments; i++) {
        hyqmom2(&moments[3*i], &xout[2*i], &wout[2*i]);
    }
    double toc = omp_get_wtime();

    printf("[OPEN_MP] Finished loop. Timer off... \n");
    return (toc - tic); 
}

/* print a usage message */
void usage(int argc, char **argv) {
    printf("usage: %s num_moments num_thread \n", argv[0]);
    return;
}

/* Set up inital moment inputs */
void init_input(float moments[], int size) {

    float one_moment[] = {0, 1, 1, 0};
    for (int i = 0; i< size * 3; i+= 3) {
        memcpy((void*)&moments[i], &one_moment, sizeof(float) * 3);
    }
}

int main(int argc, char **argv) {

    if (argc != 3) {
        usage(argc, argv);
        return 1;
    }
    int num_moments = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    float *input_moments = new float[3*num_moments];
    init_input(input_moments, num_moments);


    /* OMP */
    float omp_time = hyqmom2_openmp(input_moments, num_moments, num_threads);
    float cuda_time = hyqmom2_cuda(input_moments, num_moments);

    printf("Input size: %d \n", num_moments);
    printf("[CUDA]    Took %e s per input\n", cuda_time/num_moments);
    printf("[OPEN_MP] Took %e s per input\n", omp_time/num_moments);

    return 0;
}