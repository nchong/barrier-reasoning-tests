#!/bin/bash

if [ -z ${TAG} ]; then
  echo "Please give TAG directory, like this"
  echo "  TAG=<name> $0"
  exit 1
fi

if [ -z ${RUN} ]; then
  echo "Please give RUN parameter, like this"
  echo "  RUN=<name> $0"
  exit 1
fi

qsub -N blRace -v TAG=${TAG},TEST=blelloch,PATTERN=race_biaccess,RUN=${RUN} cx1.sh
qsub -N blAddU -v TAG=${TAG},TEST=blelloch,PATTERN=add-upsweep-*,RUN=${RUN} cx1.sh
qsub -N blOrrU -v TAG=${TAG},TEST=blelloch,PATTERN=or-upsweep-*,RUN=${RUN} cx1.sh
qsub -N blMaxU -v TAG=${TAG},TEST=blelloch,PATTERN=max-upsweep-*,RUN=${RUN} cx1.sh
qsub -N blAbsU -v TAG=${TAG},TEST=blelloch,PATTERN=abstract-upsweep-*,RUN=${RUN} cx1.sh
qsub -N blAddD -v TAG=${TAG},TEST=blelloch,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
qsub -N blOrrD -v TAG=${TAG},TEST=blelloch,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
qsub -N blMaxD -v TAG=${TAG},TEST=blelloch,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
qsub -N blAbsD -v TAG=${TAG},TEST=blelloch,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
qsub -N blAddE -v TAG=${TAG},TEST=blelloch,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
qsub -N blOrrE -v TAG=${TAG},TEST=blelloch,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
qsub -N blMaxE -v TAG=${TAG},TEST=blelloch,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
qsub -N bkRace -v TAG=${TAG},TEST=brentkung,PATTERN=race_biaccess,RUN=${RUN} cx1.sh
qsub -N bkAddD -v TAG=${TAG},TEST=brentkung,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
qsub -N bkOrrD -v TAG=${TAG},TEST=brentkung,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
qsub -N bkMaxD -v TAG=${TAG},TEST=brentkung,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
qsub -N bkAbsD -v TAG=${TAG},TEST=brentkung,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
qsub -N bkAddE -v TAG=${TAG},TEST=brentkung,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
qsub -N bkOrrE -v TAG=${TAG},TEST=brentkung,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
qsub -N bkMaxE -v TAG=${TAG},TEST=brentkung,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
qsub -N hsall  -v TAG=${TAG},TEST=.,PATTERN=hillis-steele,RUN=${RUN} cx1.sh

if [ "${ALL}" == "1" ]; then
  qsub -N allAddU -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-upsweep-*,RUN=${RUN} cx1.sh
  qsub -N allOrrU -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-upsweep-*,RUN=${RUN} cx1.sh
  qsub -N allMaxU -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-upsweep-*,RUN=${RUN} cx1.sh
  qsub -N allAbsU -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=abstract-upsweep-*,RUN=${RUN} cx1.sh
  qsub -N allAddD -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
  qsub -N allOrrD -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
  qsub -N allMaxD -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
  qsub -N allAbsD -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
  qsub -N allAddE -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
  qsub -N allOrrE -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
  qsub -N allMaxE -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
fi
