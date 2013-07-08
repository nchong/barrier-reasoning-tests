#!/bin/bash

if [ -z "${TAG}" ]; then
  echo "Please give TAG directory, like this"
  echo "  TAG=<name> $0"
  exit 1
fi

mkdir -p ${TAG}
pushd ${TAG}
date > config
FULLPATH=`pwd`
popd

dir=compact-spec
pushd $dir
mkdir ${FULLPATH}/${dir}
for ((N=4;N<=1024;N*=2)); do 
  make -f Makefile.gklee clean &&
  make -f Makefile.gklee EXTRA_CXX_FLAGS="-DN=$N" &&
  mv kernel ${FULLPATH}/${dir}/COMPACT-`printf "%04d" $N`.bc;
done
popd

for dir in blelloch brentkung koggestone; do
  pushd $dir
  mkdir ${FULLPATH}/${dir}
  for op in BINOP_ADD BINOP_OR BINOP_MAX; do
    for rwidth in 8 16 32; do
      for ((N=4;N<=1024;N*=2)); do 
        make -f Makefile.gklee clean &&
        make -f Makefile.gklee EXTRA_CXX_FLAGS="-D_SYM -DN=$N -D${op} -Drwidth=${rwidth}" &&
        mv kernel ${FULLPATH}/${dir}/${op/BINOP_/}-`printf "%04d" $N`-`printf "%02d" ${rwidth}`.bc;
      done
    done
  done
  popd
done

for dir in abstract-blelloch abstract-brentkung abstract-koggestone; do
  pushd $dir
  mkdir ${FULLPATH}/${dir}
  for ((N=4;N<=1024;N*=2)); do 
    make -f Makefile.gklee clean &&
    make -f Makefile.gklee EXTRA_CXX_FLAGS="-D_SYM -DN=$N -Drwidth=32" &&
    mv kernel ${FULLPATH}/${dir}/PAIR-`printf "%04d" $N`.bc;
  done
  popd
done  

for dir in compact-spec blelloch brentkung koggestone abstract-blelloch abstract-brentkung abstract-koggestone; do
  pushd ${FULLPATH}/${dir}
  ln -s ../../gklee-submit.sh .
  popd
done
