#!/bin/bash

DIR_ROOT=`pwd`/..
DIR_BLELLOCH=${DIR_ROOT}/tests/blelloch
DIR_BRENTKUNG=${DIR_ROOT}/tests/brentkung
DIR_HILLISSTEELE=${DIR_ROOT}/tests/hillis-steele

function populate() {
  local FR=$1
  local TO=$2
  echo ${TO}
  if [ ! -e ${TO} ]; then
    mkdir -p ${TO}
  fi
  ln -f -s ${FR}/kernel.cl ${TO}
  if [ -e `pwd`/specs ]; then
    ln -f -s `pwd`/specs ${TO}
  fi
  if [ -e ${FR}/axioms ]; then
    ln -f -s ${FR}/axioms ${TO}
  fi
}

# TAG TOP-LEVEL
if [ -z ${TAG} ]; then
  echo "Please give TAG directory, like this"
  echo "  TAG=<name> $0"
  exit 1
fi
mkdir -p ${TAG}
cd ${TAG}
date > config

# BLELLOCH
echo BLELLOCH
mkdir -p blelloch
cd blelloch
mkdir -p specs
echo -n Creating blelloch specifications...
for ((N=4; N<=2048; N*=2)); do
  python ${DIR_BLELLOCH}/genspec.py ${N} > specs/${N}_spec.h;
done
echo done

populate ${DIR_BLELLOCH} race_biaccess
for op in add max or abstract; do
  for part in upsweep downsweep endspec; do
    for width in 32 16 8; do
      if [[ "${op}" == "abstract" && "${part}" == "endspec" ]]; then
        continue;
      fi
      DIR=${op}-${part}-bv${width}
      populate ${DIR_BLELLOCH} ${DIR}
    done
  done
done
cd -

# BRENTKUNG
echo BRENTKUNG
mkdir -p brentkung
cd brentkung
mkdir -p specs
echo -n Creating brentkung specifications...
for ((N=4; N<=2048; N*=2)); do
  python ${DIR_BRENTKUNG}/genspec.py ${N} > specs/${N}_spec.h;
done
echo done

populate ${DIR_BRENTKUNG} race_biaccess
for op in add max or abstract; do
  for part in downsweep endspec; do
    for width in 32 16 8; do
      if [[ "${op}" == "abstract" && "${part}" == "endspec" ]]; then
        continue;
      fi
      DIR=${op}-${part}-bv${width}
      populate ${DIR_BRENTKUNG} ${DIR}
    done
  done
done
cd -

# HILLIS-STEELE
echo HILLIS-STEELE
mkdir -p hillis-steele
cd hillis-steele
populate ${DIR_HILLISSTEELE} .
cd -

# ALL-BLELLOCH
# echo ALL-BLELLOCH
# mkdir -p all-blelloch
# cd all-blelloch
# mkdir -p specs
# echo -n Creating blelloch specifications...
# for ((N=4; N<=2048; N*=2)); do
#   python ${DIR_BLELLOCH}/genspec.py ${N} > specs/${N}_spec.h;
# done
# echo done
#
# for op in add max or abstract; do
#   for part in upsweep downsweep endspec; do
#     for width in 32 16 8; do
#       if [[ "${op}" == "abstract" && "${part}" == "endspec" ]]; then
#         continue;
#       fi
#       DIR=${op}-${part}-bv${width}
#       populate ${DIR_BLELLOCH} ${DIR}
#     done
#   done
# done
# cd -
