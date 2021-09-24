#include <stdio.h>   
#include <stdlib.h> 

void build_array(int *b) {
    b[0] = 0;
    b[1] = 0;
    b[2] = 0;
    return;
}

int main() {
    int arr[5] = { 1, 2, 4, 5, 6 };
    int *ptr = arr;
    build_array(ptr);
    for (int k = 0; k < 5; k++) {
        printf("%d\n", ptr[k]);
    }
    return 0;
}


