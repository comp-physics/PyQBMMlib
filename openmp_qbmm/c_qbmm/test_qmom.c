#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* void hyqmom2(float m[], float x[], float w[]); */
/* void chyqmom4(float m[], float x[], float y[]); */
/* void chyqmom4(float m[], float x[], float y[], float w[]); */
/* void chyqmom4(float m[], float *x[], float *y[], float *w[]); */


/* void hyqmom2(float *mom, float *x, float *w) { */
/*     float bx, d2, c2; */

/*     w[0] = mom[0] / 2.0; */
    /* w[1] = w[0]; */

    /* bx = mom[1] / mom[0]; */
    /* d2 = mom[2] / mom[0]; */
    /* c2 = d2 - pow(bx,2.0); */

    /* x[0] = bx - sqrt(c2); */
    /* x[1] = bx + sqrt(c2); */
}

/* void hyqmom2(float mom[], float x[], float w[]) { */
/*     float bx, d2, c2; */

/*     w[0] = mom[0] / 2.0; */
/*     w[1] = w[0]; */

/*     bx = mom[1] / mom[0]; */
/*     d2 = mom[2] / mom[0]; */
/*     c2 = d2 - pow(bx,2.0); */

/*     x[0] = bx - sqrt(c2); */
/*     x[1] = bx + sqrt(c2); */
/* } */


/* void chyqmom4(float mom[], float xout[], float yout[]) { */
void chyqmom4(float mom[], float *xout[], float *yout[], float *wout[]) {
    int i;
    int n1d = 2;
    int n2d = 4;
    float mom00, mom10, mom01, mom20, mom11, mom02;
    float bx, by, d20, d11, d02, c20, c11, c02;
    float mu2, mu2avg;
    float yp21, yp22, rho21, rho22;
    float xp[n1d], rho[n1d], yf[n1d];
    float xp3[n1d], rh3[n1d];
    /* float wout[n2d]; */
    /* float xout[n2d], yout[n2d], wout[n2d]; */

    mom00 = mom[0];
    mom10 = mom[1];
    mom01 = mom[2];
    mom20 = mom[3];
    mom11 = mom[4];
    mom02 = mom[5];

    bx = mom10 / mom00;
    by = mom01 / mom00;
    d20 = mom20 / mom00;
    d11 = mom11 / mom00;
    d02 = mom02 / mom00;

    c20 = d20 - pow(bx,2);
    c11 = d11 - bx * by;
    c02 = d02 - pow(by,2);

    float M1[] = { 1., 0., c20 };

    hyqmom2(M1,xp,rho);
    for (i=0; i<n1d; i++) {
        yf[i] = c11 * xp[i] / c20;
    }

    mu2avg = c02;
    for (i=0; i<n1d; i++) {
        mu2avg -= rho[i]*pow(yf[i],2.);
    }

    if (mu2avg <= 0.) {
        mu2avg = 0.;
    }

    mu2 = mu2avg;
    float M3[] = { 1., 0., mu2 };
    hyqmom2(M3,xp3,rh3);

    yp21 = xp3[0];
    yp22 = xp3[1];
    rho21 = rh3[0];
    rho22 = rh3[1];

    *wout[0] = mom00 * rho[0] * rho21;
    *wout[1] = mom00 * rho[0] * rho22;
    /* wout[2] = mom00 * rho[1] * rho21; */
    /* wout[3] = mom00 * rho[1] * rho22; */

    /* xout[0] = bx * xp[0]; */
    /* xout[1] = bx * xp[0]; */
    /* xout[2] = bx * xp[1]; */
    /* xout[3] = bx * xp[1]; */

    /* yout[0] = by + yf[0] + yp21; */
    /* yout[1] = by + yf[0] + yp22; */
    /* yout[2] = by + yf[1] + yp21; */
    /* yout[3] = by + yf[1] + yp22; */

    return;
}

/* void chyqmom4(float mom[], float xout[], float yout[], float wout[]) { */
/*     int i; */
/*     int n1d = 2; */
/*     float mom00, mom10, mom01, mom20, mom11, mom02; */
/*     float bx, by, d20, d11, d02, c20, c11, c02; */
/*     float mu2, mu2avg; */
/*     float yp21, yp22, rho21, rho22; */
/*     float xp[n1d], rho[n1d], yf[n1d]; */
/*     float xp3[n1d], rh3[n1d]; */

/*     mom00 = mom[0]; */
/*     mom10 = mom[1]; */
/*     mom01 = mom[2]; */
/*     mom20 = mom[3]; */
/*     mom11 = mom[4]; */
/*     mom02 = mom[5]; */

/*     bx = mom10 / mom00; */
/*     by = mom01 / mom00; */
/*     d20 = mom20 / mom00; */
/*     d11 = mom11 / mom00; */
/*     d02 = mom02 / mom00; */

/*     c20 = d20 - pow(bx,2); */
/*     c11 = d11 - bx * by; */
/*     c02 = d02 - pow(by,2); */

/*     float M1[] = { 1., 0., c20 }; */

/*     hyqmom2(M1,xp,rho); */
/*     for (i=0; i<n1d; i++) { */
/*         yf[i] = c11 * xp[i] / c20; */
/*     } */

/*     mu2avg = c02; */
/*     for (i=0; i<n1d; i++) { */
/*         mu2avg -= rho[i]*pow(yf[i],2.); */
/*     } */

/*     if (mu2avg <= 0.) { */
/*         mu2avg = 0.; */
/*     } */

/*     mu2 = mu2avg; */
/*     float M3[] = { 1., 0., mu2 }; */
/*     hyqmom2(M3,xp3,rh3); */

/*     yp21 = xp3[0]; */
/*     yp22 = xp3[1]; */
/*     rho21 = rh3[0]; */
/*     rho22 = rh3[1]; */

/*     wout[0] = mom00 * rho[0] * rho21; */
/*     wout[1] = mom00 * rho[0] * rho22; */
/*     wout[2] = mom00 * rho[1] * rho21; */
/*     wout[3] = mom00 * rho[1] * rho22; */

/*     xout[0] = bx * xp[0]; */
/*     xout[1] = bx * xp[0]; */
/*     xout[2] = bx * xp[1]; */
/*     xout[3] = bx * xp[1]; */

/*     yout[0] = by + yf[0] + yp21; */
/*     yout[1] = by + yf[0] + yp22; */
/*     yout[2] = by + yf[1] + yp21; */
/*     yout[3] = by + yf[1] + yp22; */

/*     return; */
/* } */


int main(int argc, char* argv[]){
    int i,j;
    double tic, toc;

    int n1d = 2;
    float *mom1D;
    float *x1D;
    float *w1D;

    /* int n2d = 4; */
    /* float *mom2D; */
    /* float *x2D; */
    /* float *y2D; */ 
    /* float *w2D; */ 

    mom1D = malloc(3*sizeof(float));
    x1D = malloc(n1d*sizeof(float));
    w1D = malloc(n1d*sizeof(float));
    

    if(argc<=1) {
        printf("No arguments passed, abort");
        exit(1);
    }
    int nx = atoi(argv[1]);

    // Test HyQMOM2 timing
    mom1D[0] = 1.;
    mom1D[1] = 0.001;
    mom1D[2] = 0.03;

    for (i=0;i<3;i++) {
        printf("mom[%d] = %f\n",i,mom1D[i]);
    }
    hyqmom2(&mom1D,&x1D,&w1D);
    for (i=0;i<n1d;i++) {
        printf("x[%d],w[%d] = %f, %f\n",x1D[i],w1D[i]);
    }

    /* tic = omp_get_wtime(); */
    /* #pragma omp parallel for */
    /* for (i=0; i<nx; i++){ */
    /*     hyqmom2(mom1D,x1D,w1D); */
    /* } */
    /* toc = omp_get_wtime(); */
    /* printf("%.0e iterations of  HyQMOM2 on %d thread(s) took %f seconds\n", (float)nx, omp_get_max_threads(), toc-tic); */

    // Test CHyQMOM4 timing
    /* mom2D[0] = 1.; */
    /* mom2D[1] = 1.; */
    /* mom2D[2] = 1.; */
    /* mom2D[3] = 1.01; */
    /* mom2D[4] = 1.; */
    /* mom2D[5] = 1.01; */
    /* tic = omp_get_wtime(); */
    /* #pragma omp parallel for */
    /* for (i=0; i<nx; i++){ */
    /*     chyqmom4(mom2D,&x2D,&y2D,&w2D); */
    /*     /1* chyqmom4(mom2D,x2D,y2D,w2D); *1/ */
    /* } */
    /* toc = omp_get_wtime(); */
    /* printf("%.0e iterations of CHyQMOM4 on %d thread(s) took %f seconds\n", (float)nx, omp_get_max_threads(), toc-tic); */

    return 0;
}
