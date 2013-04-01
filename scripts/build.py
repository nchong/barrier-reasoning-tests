import getopt
import subprocess
import sys
import os

GPUVERIFY_INSTALL_DIR = os.environ.get('GPUVERIFY_INSTALL_DIR')
AXIOMS_DIR = 'axioms'
KERNEL     = 'kernel.cl'

class BINOP(object):
  ADD      = 'BINOP_ADD'
  MAX      = 'BINOP_MAX'
  OR       = 'BINOP_OR'
  ABS      = 'BINOP_ABSTRACT'
  INTERVAL = 'BINOP_INTERVAL'

class CHECK(object):
  RACE      = 'CHECK_RACE'
  BI        = 'CHECK_BI'
  BI_ACCESS = 'CHECK_BI_ACCESS'

class PART(object):
  UPSWEEP   = 'INC_UPSWEEP'
  DOWNSWEEP = 'INC_DOWNSWEEP'
  ENDSPEC   = 'INC_ENDSPEC'

class SPEC(object):
  THREAD   = 'SPEC_THREADWISE'
  ELEMENT  = 'SPEC_ELEMENTWISE'
  INTERVAL = 'SPEC_INTERVAL'

class Options(object):
  N = 4
  op = BINOP.ADD
  width = 32
  verbose = False
  flags = ""
  parts = []
  spec = SPEC.ELEMENT
  boogie_file = None
  memout = 8000 # megabytes
  timeout = 3600 # seconds
  relentless = False
  repeat = 0
  mkbpl = False
  specs_dir = 'specs'

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
  print '  --spec=X'
  print '  --specs-dir=X'
  print '  --boogie-file=X'
  print '  --memout=X'
  print '  --timeout=X'
  print '  --relentless'
  print '  --repeat='
  print '  --stop-at-bpl'
  return 0

def error(msg):
  print 'ERROR: %s' % msg
  return 1

def main(doit,header=None,argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  try:
    opts, args = getopt.getopt(argv[1:],'h',
      ['verbose','help',
       'op=','width=','flags=',
       'upsweep','downsweep','endspec',
       'boogie-file=','memout=','timeout=','relentless','repeat=',
       'spec=','stop-at-bpl', 'specs-dir=',
      ])
  except getopt.GetoptError:
    return error('error parsing options; try -h')
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
      elif op == 'interval': Options.op = BINOP.INTERVAL
      else: return error('operator [%s] not recognised' % a)
    if o == "--width":
      try:
        width = int(a)
      except ValueError:
        return error('width must be an integer')
      if width not in [8,16,32]:
        return error('width must be one of 8, 16, 32')
      Options.width = width
    if o == "--spec":
      spec = a.lower()
      if spec == 'element': Options.spec = SPEC.ELEMENT
      elif spec == 'thread': Options.spec = SPEC.THREAD
      elif spec == 'interval': Options.spec = SPEC.INTERVAL
      else: return error('spec must be one of element, thread')
    if o == "--flags":
      Options.flags = a
    if o == '--upsweep':
      Options.parts.append(PART.UPSWEEP)
    if o == '--downsweep':
      Options.parts.append(PART.DOWNSWEEP)
    if o == '--endspec':
      Options.parts.append(PART.ENDSPEC)
    if o == '--boogie-file':
      Options.boogie_file = a
    if o == '--stop-at-bpl':
      Options.mkbpl = True
    if o == '--memout':
      try:
        Options.memout = int(a)
      except ValueError:
        return error('bad memout [%s] given' % a)
    if o == '--timeout':
      try:
        Options.timeout = int(a)
      except ValueError:
        return error('bad timeout [%s] given' % a)
    if o == "--relentless":
      Options.relentless = True
    if o == "--specs-dir":
      Options.specs_dir = a
    if o == "--repeat":
      try:
        Options.repeat = int(a)
      except ValueError:
        return error('bad repeat [%s] given' % a)
  if not GPUVERIFY_INSTALL_DIR:
    return error('Could not find GPUVERIFY_INSTALL_DIR environment variable')
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
 #if PART.ENDSPEC in Options.parts and Options.op == BINOP.ABS:
 #  return error('endspec for abstract operator is not supported')
  code = doit()
  if Options.relentless: # keep going for more!
    while code == 0:
      Options.N = 2 * Options.N
      code = doit()
  while code == 0 and Options.repeat > 0:
    Options.repeat -= 1
    code = doit()
  return code

def buildcmd(checks,extraflags=[]):
  cmd = [ GPUVERIFY_INSTALL_DIR + '/gpuverify',
          '--silent',
          '--time-as-csv=%s' % fname(),
          '--testsuite',
          '--no-infer',
          '--no-source-loc-infer',
          '--only-intra-group',
          '--timeout=%d' % Options.timeout,
          '-I%s' % Options.specs_dir,
          '-DN=%d' % Options.N,
          '-D%s' % Options.op,
          '-Ddwidth=32',
          '-Drwidth=%d' % Options.width,
        ]
  if Options.memout > 0:
    cmd.append('--memout=%d' % Options.memout)
  if PART.ENDSPEC in Options.parts:
    cmd.append('-D%s' % Options.spec)
  cmd.extend(['-D%s' % x for x in Options.parts])
  cmd.extend(['-D%s' % x for x in checks])
  if Options.boogie_file:
    cmd.append('--boogie-file=%s' % Options.boogie_file)
  elif Options.op == BINOP.ABS and CHECK.BI in checks:
    if PART.UPSWEEP in Options.parts: bpl = 'upsweep'
    elif PART.DOWNSWEEP in Options.parts: bpl = 'downsweep'
    cmd.append('--boogie-file=%s/%s%d.bpl' % (AXIOMS_DIR,bpl,Options.width))
  cmd.extend(extraflags)
  cmd.append(Options.flags)
  if Options.mkbpl: cmd.append('--stop-at-bpl')
  cmd.append(KERNEL)
  return cmd

def fname(suffix=None):
  def aux(x): return x.split('_')[1].lower()
  op = aux(Options.op)
  part = '_'.join([aux(x) for x in Options.parts])
  if part == 'upsweep_downsweep': part = 'race-biacc'
  if Options.width == 32: ty = 'uint'
  elif Options.width == 16: ty = 'ushort'
  elif Options.width == 8: ty = 'uchar'
  else: assert False
  if suffix: return '%04d-%s-%s-%s.%s' % (Options.N,op,part,ty,suffix)
  else: return '%04d-%s-%s-%s' % (Options.N,op,part,ty)

def run(cmd):
  if Options.verbose: print ' '.join(cmd)
  code = subprocess.call(cmd)
  if code == 0 and Options.mkbpl:
    os.rename('kernel.bpl', fname('bpl'))
  return code

def build_and_run(checks,extraflags=[]):
  cmd = buildcmd(checks,extraflags)
  return run(cmd)
