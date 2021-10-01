#include <cstdio>
#include <iostream>
#include <cstring>
#include <cassert>
void init_input_col_major(float moments[], int size);
void init_input_row_major(float moments[], int size) ;

float run_coal(const float moment[], const int size, float x_out[], float w_out[]);
float run_naive(const float moment[], const int size, float x_out[], float w_out[]);

