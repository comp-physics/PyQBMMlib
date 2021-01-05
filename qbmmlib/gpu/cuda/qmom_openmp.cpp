#include <cmath>

#include <omp.h>
#include <stdbool.h>

#include "main.hpp"

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

void chyqmom4(float mom[], float xout[], float yout[], float wout[]) {
    int i;
    int n1d = 2;
    int n2d = 4;
    float mom00, mom10, mom01, mom20, mom11, mom02;
    float bx, by, d20, d11, d02, c20, c11, c02;
    float mu2, mu2avg;
    float yp21, yp22, rho21, rho22;
    float xp[n1d], rho[n1d], yf[n1d];
    float xp3[n1d], rh3[n1d];

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

    wout[0] = mom00 * rho[0] * rho21;
    wout[1] = mom00 * rho[0] * rho22;
    wout[2] = mom00 * rho[1] * rho21;
    wout[3] = mom00 * rho[1] * rho22;

    xout[0] = bx + xp[0];
    xout[1] = bx + xp[0];
    xout[2] = bx + xp[1];
    xout[3] = bx + xp[1];

    yout[0] = by + yf[0] + yp21;
    yout[1] = by + yf[0] + yp22;
    yout[2] = by + yf[1] + yp21;
    yout[3] = by + yf[1] + yp22;

    return;
}

float qmom_openmp(float moments[], int num_moments, int nthread, 
                    float xout[], float yout[], float wout[])
{
    omp_set_num_threads(nthread);

    printf("[OPEN_MP] starting %d thread(s) \n", omp_get_max_threads());
    printf("[OPEN_MP] starting loop. Timer on... \n");

    double tic = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0; i<num_moments; i++) {
        chyqmom4(&moments[6*i], &xout[4*i], &yout[4*i], &wout[4*i]);
    }
    double toc = omp_get_wtime();

    printf("[OPEN_MP] Finished loop. Timer off... \n");
    return (toc - tic)*1e3; // convert to miliseconds 
}

float qmom_naive(float moments[], int num_moments,
                    float xout[], float yout[], float wout[])
{

    printf("[NAIVE] starting loop. Timer on... \n");

    double tic = omp_get_wtime();
    for (int i=0; i<num_moments; i++) {
        chyqmom4(&moments[6*i], &xout[4*i], &yout[4*i], &wout[4*i]);
    }
    double toc = omp_get_wtime();
    printf("[NAIVE] Finished loop. Timer off... \n");
    return (toc - tic)*1e3;
}