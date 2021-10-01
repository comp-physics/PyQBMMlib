#include "main.hpp"

/* Set up inital moment inputs row major style*/
void init_input_col_major(float moments[], int size) {

    float one_moment[] = {1, 0, 1};
    for (int i = 0; i< size*3; i+=3) {
        memcpy((void*)&moments[i], &one_moment, sizeof(float) * 3);
    }
}
/* Set up inital moment inputs column major style*/
void init_input_row_major(float moments[], int size) {
    float one_moment[] = {1, 0, 1};
    for (int i = 0; i< size; i++) {
        moments[i] = one_moment[0];
        moments[i+size] = one_moment[1];
        moments[i+2*size] = one_moment[2];
    }
}

