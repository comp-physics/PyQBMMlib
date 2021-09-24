#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

float f(int n) { 
    float q = 0;
    for(int i = 1; i <= n; i++){
        q += i ;
    }
    return q;
} 

int main(int argc, char** argv){
    if(argc<=1) {
        printf("No arguments passed, abort");
        exit(1);
    }
    int nx = atoi(argv[1]);

    double tic = omp_get_wtime();
    float partial_Sum = 0.;
    #pragma omp parallel for
    for(int i = 1; i <= nx; i++){
        partial_Sum += (float)i + f(10);
    }
    double toc = omp_get_wtime();

    printf("%.0e iterations of test_omp on %d thread(s) took %f seconds\n", (float)nx, omp_get_max_threads(), toc-tic);
    return 0;
}
