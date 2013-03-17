#!/usr/bin/env python

import os
import chkall
import chkbi
import chkrace_biaccess
import sys

cwd = os.getcwd().split(os.sep)[-1]
print cwd
if cwd == 'hillis-steele':
  args = [ 'dummy', '--upsweep', '--relentless', '2' ]
  main = chkall.main_wrapper
elif cwd == 'race_biaccess':
  args = [ 'dummy', '--upsweep', '--downsweep', '--relentless', '4' ]
  main = chkrace_biaccess.main_wrapper
else: # otherwise blelloch or brentkung subdirectory
  op, part, width = cwd.split('-')
  width = int(width[2:])
  assert op in [ 'add', 'or', 'max', 'abstract' ]
  assert part in [ 'upsweep', 'downsweep', 'endspec' ]
  assert width in [ 8,16,32 ]
  args = [ 'dummy', '--op=%s' % op, '--%s' % part, '--width=%d' % width, '--relentless', '4' ]
  main = chkbi.main_wrapper
  if os.environ.get('CHKALL'):
    print "Forcing check all"
    main = chkall.main_wrapper

sys.stdout.flush()
main(args)
