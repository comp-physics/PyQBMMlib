name=$1
inputFile=$2

if [ -f submit.sh ]
then
    rm submit.sh
fi

echo "#!/bin/bash

#MSUB -N tunnel
#MSUB -l nodes=1
#MSUB -q pbatch
#MSUB -o ${name}.out
#MSUB -e ${name}.err
#MSUB -l walltime=00:15:00
#MSUB -A uiuc

python driver.py $inputFile

">>submit.sh
