#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=1:mem=8000mb

echo "-------------------------------------------------"
echo ${TAG}
echo "-------------------------------------------------"
cat /proc/cpuinfo

cd /work/nyc04/regression
export GPUVERIFY_INSTALL_DIR=`pwd`/GPUVerifyInstall
export PATH=`pwd`/local/bin:$PATH
export LD_LIBRARY_PATH=`pwd`/local/lib64:`pwd`/local/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
module load python/2.7.3

cd ${PBS_O_WORKDIR}
cd ${TAG}
for DIR in `find ${TEST} -maxdepth 1 -name "${PATTERN}" -print | sort`; do
  echo "-------------------------------------------------"
  echo ${DIR}
  cd ${DIR}
  ${PBS_O_WORKDIR}/submit.py > >(tee ${RUN}) 2> >(tee err.log >&2)
  cd -
done
