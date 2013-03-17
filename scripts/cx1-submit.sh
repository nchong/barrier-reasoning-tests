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

qsub -v TAG=${TAG},TEST=blelloch,PATTERN=race_biaccess,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=add-upsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=or-upsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=max-upsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=abstract-upsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=blelloch,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=race_biaccess,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=brentkung,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
qsub -v TAG=${TAG},TEST=.,PATTERN=hillis-steele,RUN=${RUN} cx1.sh

if [ -z ${ALL} ]; then
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-upsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-upsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-upsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=abstract-upsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-downsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-downsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-downsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=abstract-downsweep-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=add-endspec-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=or-endspec-*,RUN=${RUN} cx1.sh
  qsub -v CHKALL=1,TAG=${TAG},TEST=all-blelloch,PATTERN=max-endspec-*,RUN=${RUN} cx1.sh
fi
