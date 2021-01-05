#!/bin/bash

## Provide me an input argument
## for how large of a grid to make
## e.g. ./build.sh 100000

file_out='test_qmom'
# file_out='test_omp'
# file_out='test_omp2'
rm -f $file_out
if command -v clang &> /dev/null
then
    ## This clang command will work on MacOS, though
    ## you may have to install OpenMP via
    ## $ brew install libomp
    # clang -Xpreprocessor -fopenmp -O0 test_qmom.c -lomp -o $file_out
    clang -Xpreprocessor -fopenmp -O3 $file_out.c -lomp -o $file_out
else
    # try gcc
    gcc -fopenmp -O3 $file_out.c -lgomp -lm -o $file_out
fi

# Loop through different numbers of OpenMP threads
if [[ -f "$file_out" ]]; then
    # for q in {1..4}; do
    for q in {1..1}; do
        export OMP_NUM_THREADS=$q
        ./$file_out $1
    done
fi
