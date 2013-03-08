import getopt
import subprocess
import sys

#GPUVERIFY_INSTALL_DIR = '/Users/nafe/work/gpuverify/GPUVerifyInstall'
GPUVERIFY_INSTALL_DIR = '/work/nyc04/regression/GPUVerifyInstall'
SPECS_DIR  = 'specs'
AXIOMS_DIR = 'axioms'
KERNEL     = 'kernel.cl'

class BINOP(object):
  ADD = 'BINOP_ADD'
  MAX = 'BINOP_MAX'
  OR  = 'BINOP_OR'
  ABS = 'BINOP_ABSTRACT'

class CHECK(object):
  RACE      = 'CHECK_RACE'
  BI        = 'CHECK_BI'
  BI_ACCESS = 'CHECK_BI_ACCESS'

class PART(object):
  UPSWEEP   = 'INC_UPSWEEP'
  DOWNSWEEP = 'INC_DOWNSWEEP'
  ENDSPEC   = 'INC_ENDSWEEP'

class Options(object):
  N = 4
  op = BINOP.ADD
  width = 32
  verbose = False
  flags = ""
  parts = []

def ispow2(x):
  return x != 0 and ((x & (x-1)) == 0)

def help(progname,header=None):
  if header:
    print 'SYNOPSIS: %s' % header
    print
  print 'USAGE: %s [options] n' % progname
  print
  print '  -h             Display this message'
  print '  --verbose      Show commands to run'
  print '  --op=X         Choose binary operator'
  print '  --width=X      Choose bitwidth'
  print '  --flags=X      Flags for GPUVerify'
  print '  --upsweep'
  print '  --downsweep'
  print '  --endspec'
  return 0

def error(msg):
  print 'ERROR: %s' % msg
  return 1

def main(doit,header=None,argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  opts, args = getopt.getopt(argv[1:],'h',
    ['verbose','op=','width=','flags=',
     'upsweep','downsweep','endspec',
    ])
  for o, a in opts:
    if o in ('-h','--help'):
      return help(progname,header)
    if o == "--verbose":
      Options.verbose = True
    if o == "--op":
      op = a.lower()
      if op == 'add':        Options.op = BINOP.ADD
      elif op == 'max':      Options.op = BINOP.MAX
      elif op == 'or':       Options.op = BINOP.OR
      elif op == 'abstract': Options.op = BINOP.ABS
      else: return error('')
    if o == "--width":
      try:
        width = int(a)
      except ValueError:
        return error('width must be an integer')
      if width not in [8,16,32]:
        return error('width must be one of 8, 16, 32')
    if o == "--flags":
      Options.flags = a
    if o == '--upsweep':
      Options.parts.append(PART.UPSWEEP)
    if o == '--downsweep':
      Options.parts.append(PART.DOWNSWEEP)
    if o == '--endspec':
      Options.parts.append(PART.ENDSPEC)
  if len(args) != 1:
    return error('number of elements not specified')
  try:
    Options.N = int(args[0])
  except ValueError:
    return error('invalid value given for n')
  if not ispow2(Options.N):
    return error('n must be a power of two')
  if len(Options.parts) == 0:
    return error('specify parts to check')
  if PART.ENDSPEC in Options.parts and Options.op == BINOP.ABS:
    return error('endspec for abstract operator is not supported')
  doit()
  return 0

def buildcmd(checks,extraflags=[]):
  cmd = [ GPUVERIFY_INSTALL_DIR + '/gpuverify',
          '--time',
          '--testsuite',
          '--no-infer',
          '--no-source-loc-infer',
          '--only-intra-group',
          '-I%s' % SPECS_DIR,
          '-DN=%d' % Options.N,
          '-D%s' % Options.op,
          '-Ddwidth=%d' % Options.width,
          '-Drwidth=%d' % Options.width,
        ]
  cmd.extend(['-D%s' % x for x in Options.parts])
  cmd.extend(['-D%s' % x for x in checks])
  if Options.op == BINOP.ABS and CHECK.BI in checks:
    if PART.UPSWEEP in Options.parts: bpl = 'upsweep'
    elif PART.DOWNSWEEP in Options.parts: bpl = 'downsweep'
    cmd.append('--boogie-file=%s/%s%d.bpl' % (AXIOMS_DIR,bpl,Options.width))
  cmd.extend(extraflags)
  cmd.append(Options.flags)
  cmd.append(KERNEL)
  return cmd

def run(cmd):
  if Options.verbose: print ' '.join(cmd)
  subprocess.call(cmd)

def build_and_run(checks,extraflags=[]):
  cmd = buildcmd(checks,extraflags)
  run(cmd)
