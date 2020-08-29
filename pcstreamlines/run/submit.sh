#!/bin/bash

#MSUB -N tunnel
#MSUB -l nodes=1
#MSUB -q pbatch
#MSUB -o streamlines-15.out
#MSUB -e streamlines-15.err
#MSUB -l walltime=00:30:00
#MSUB -A uiuc

python driver.py ../inputs/actii_3D_freestream_line_15.yaml


