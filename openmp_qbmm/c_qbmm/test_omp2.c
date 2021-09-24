#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>

float f(int n) { 
    float q = 0;
    for(int i = 1; i <= n; i++){
        q += i ;
    }
    return q;
} 

int main(int argc, char** argv){
    double toc, tic;
    double integralSum = 0;

    if(argc<=1) {
        printf("No arguments passed, abort");
        exit(1);
    }
    int parts = atoi(argv[1]);

    float from = 0.0f;
    float to = 2.0f;
    float step = ((to - from) / (float)parts);
    float x = (from + (step / 2.0f));

    tic = omp_get_wtime();
    int i;
    float partial_Sum = 0;
    #pragma omp parallel for reduction(+:integralSum)
    for (i = 1; i < (parts+1); ++i) {
        integralSum = integralSum + (step * fabs(pow((x + (step * i)),2) + 4));
    }
    toc = omp_get_wtime();

    printf("%.0e iterations of test_omp2 on %d thread(s) took %f seconds\n", (float)parts, omp_get_max_threads(), toc-tic);
    return 0;
}
