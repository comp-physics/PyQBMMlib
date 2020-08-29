#!/bin/bash

fileRoot="../inputs/actii_3D_freestream_line_"

for ((line=1;line<=51;line=line+1))
do
    name="streamlines-"${line}
    inputFile=${fileRoot}${line}".yaml"
    sh writeSubmissionScript.sh $name $inputFile
    echo ""
    echo "Running streamlines driver with input file "$inputFile
    echo ""
    msub submit.sh
    #python driver.py ${inputFile}
done

