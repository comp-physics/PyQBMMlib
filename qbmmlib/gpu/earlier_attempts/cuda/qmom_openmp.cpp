#include <cmath>

#include <omp.h>
#include <stdbool.h>

#include "main.hpp"


float sum_pow_cpp(float rho[], float yf[], float n, int len){
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += rho[i] * pow(yf[i], n);
    }
    return sum;
}

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

void hyqmom3(float mom[], float x[], float w[]) {
    float bx, d2, d3, d4;
    bx = mom[1] / mom[0];
    d2 = mom[2] / mom[0];
    d3 = mom[3] / mom[0];
    d4 = mom[4] / mom[0];

    float c2, c3, c4;
    c2 = d2 - bx * bx;
    c3 = d3 - 3 * bx * d2 + 2 * bx * bx * bx;
    c4 = d4 - 4 * bx * d3 + 6 * (bx * bx) * d2 - 3 * pow(bx, 4);

    float scale = sqrt(c2);
    float q = c3 / sqrt(c2) / c2;
    float eta = c4 / c2 / c2;

    float *xps = new float[3];
    xps[0] = (q - sqrt(4 * eta - 3 * q * q)) / 2.0;
    xps[1] = 0.0;
    xps[2] = (q + sqrt(4 * eta - 3 * q * q)) / 2.0;

    float dem = 1.0 / sqrt(4 * eta - 3 * q * q);
    float prod = -xps[0] * xps[2];

    float *rho = new float[3];
    rho[0] = -dem / xps[0];
    rho[1] = 1 - 1 / prod;
    rho[2] = dem / xps[2];
    float srho = rho[0] + rho[1] + rho[2];

    float scales = 0;
    for (int i = 0; i < 3; i++) {
        rho[i] /= srho;
        scales += rho[i] * xps[i] * xps[i];
    }
    scales /= srho;
    for (int i = 0; i < 3; i++) {
        x[i] = xps[i] * scale / sqrt(scales) + bx;
        w[i] = mom[0] * rho[i];
    }
    delete[] rho;
    delete[] xps;

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

void chyqmom9(float mom[], float xout[], float yout[], float wout[]) {

    float mom00 = mom[0];
    float mom10 = mom[1];
    float mom01 = mom[2];
    float mom20 = mom[3];
    float mom11 = mom[4];
    float mom02 = mom[5];
    float mom30 = mom[6];
    float mom03 = mom[7];
    float mom40 = mom[8];
    float mom04 = mom[9];

    float bx = mom10 / mom00;
    float by = mom01 / mom00;
    float d20 = mom20 / mom00;
    float d11 = mom11 / mom00;
    float d02 = mom02 / mom00;
    float d30 = mom30 / mom00;
    float d03 = mom03 / mom00;
    float d40 = mom40 / mom00;
    float d04 = mom04 / mom00;

    float c20 = d20 - bx * bx;
    float c11 = d11 - bx * by;
    float c02 = d02 - by * by;
    float c30 = d30 - 3.0 * bx * d20 + 2.0 * bx * bx * bx;
    float c03 = d03 - 3.0 * by * d02 + 2.0 * by * by * by;
    float c40 = d40 - 4.0 * bx * d30 + 6 * (bx * bx) * d20 - 3.0 * pow(bx, 4);
    float c04 = d04 - 4.0 * by * d03 + 6 * (by * by) * d02 - 3.0 * pow(by, 4);

    float M1[5] = {1, 0, c20, c30, c40};
    float *xp = new float[3];
    float *rho = new float[3];
    hyqmom3(M1, xp, rho);

    float yf[3];
    for (int i = 0; i < 3; i++) {
        yf[i] = c11 * xp[i] / c20;
    }
    float mu2 = c02 - sum_pow_cpp(rho, yf, 2, 3);

    float q = (c03 - sum_pow_cpp(rho, yf, 3, 3)) / pow(mu2, (3.0 / 2.0));
    float eta =  (c04 - sum_pow_cpp(rho, yf, 4, 3) - 6 *sum_pow_cpp(rho, yf, 2, 3) * mu2)/ pow(mu2, 2);
    float mu3 = q * pow(mu2, 3.0/2.0);
    float mu4 = eta * mu2 * mu2;

    float M3[5] = {1, 0, mu2, mu3, mu4};
    float *xp3 = new float[3];
    float *rh3 = new float[3];
    hyqmom3(M3, xp3, rh3);

    float yp21 = xp3[0];
    float yp22 = xp3[1];
    float yp23 = xp3[2];
    float rho21 = rh3[0];
    float rho22 = rh3[1];
    float rho23 = rh3[2];

    wout[0] = mom00 * rho[0] * rho21;
    wout[1] = mom00 * rho[0] * rho22;
    wout[2] = mom00 * rho[0] * rho23;
    wout[3] = mom00 * rho[1] * rho21;
    wout[4] = mom00 * rho[1] * rho22;
    wout[5] = mom00 * rho[1] * rho23;
    wout[6] = mom00 * rho[2] * rho21;
    wout[7] = mom00 * rho[2] * rho22;
    wout[8] = mom00 * rho[2] * rho23;

    xout[0] = xp[0] + bx;
    xout[1] = xp[0] + bx;
    xout[2] = xp[0] + bx;
    xout[3] = xp[1] + bx;
    xout[4] = xp[1] + bx;
    xout[5] = xp[1] + bx;
    xout[6] = xp[2] + bx;
    xout[7] = xp[2] + bx;
    xout[8] = xp[2] + bx;

    yout[0] = yf[0] + yp21;
    yout[1] = yf[0] + yp22;
    yout[2] = yf[0] + yp23;
    yout[3] = yf[1] + yp21;
    yout[4] = yf[1] + yp22;
    yout[5] = yf[1] + yp23;
    yout[6] = yf[2] + yp21;
    yout[7] = yf[2] + yp22;
    yout[8] = yf[2] + yp23;

    delete[] xp3;
    delete[] rh3;
    delete[] rho;
    delete[] xp;
}

float chyqmom4_omp(float moments[], int num_moments, int nthread, 
                    float xout[], float yout[], float wout[])
{
    float *moment_col_major = new float[num_moments*6];
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < num_moments; col ++) {
            moment_col_major[col * 6 + row] = moments[row * num_moments + col];
        }
    }
    omp_set_num_threads(nthread);

    double tic = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0; i<num_moments; i++) {
        chyqmom4(&moments[6*i], &xout[4*i], &yout[4*i], &wout[4*i]);
    }
    double toc = omp_get_wtime();
    
    delete[] moment_col_major;

    return (toc - tic) * 1e3; // covert to milliseconds 
}


float chyqmom9_omp(float moments[], int num_moments, int nthread, 
                    float xout[], float yout[], float wout[])
{
    float *moment_col_major = new float[num_moments*10];
    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < num_moments; col ++) {
            moment_col_major[col * 10 + row] = moments[row * num_moments + col];
        }
    }
    omp_set_num_threads(nthread);

    double tic = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0; i<num_moments; i++) {
        chyqmom9(&moments[10*i], &xout[9*i], &yout[9*i], &wout[9*i]);
    }
    double toc = omp_get_wtime();
    
    delete[] moment_col_major;

    return (toc - tic) * 1e3; // covert to milliseconds 
}