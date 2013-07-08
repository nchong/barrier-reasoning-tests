#!/bin/bash
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mem=16000mb

cd /work/nyc04/trunk/
source sourceme.sh
cd $PBS_O_WORKDIR
echo $FILE >&2
time gklee -max-memory=16000 -no-output -output-dir=klee-$FILE $FILE &> $FILE.test
echo $FILE >&2
