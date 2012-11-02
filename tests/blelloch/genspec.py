#!/usr/bin/env python

from math import log
import sys
try:
  from jinja2 import Environment, PackageLoader
except ImportError:
  print 'Error: This script requires the jinja2 template library'
  print 'See here for instructions for installation'
  print 'http://jinja.pocoo.org/docs/intro/#installation'
  exit(-1)

def log2(x):
  return int(log(x,2))

def summation(terms,f):
  return reduce(lambda x,y: '%s(%s,%s)' % (f,x,y), terms)

def upsweep_pattern(N,rhs):
  terms = [ 'len[x]' ]
  def lhs(off):
    return '((deq(offset, %d) & isvertex(x,offset)) | (dlt(%d,offset) & stopped(x,%d)))' % (off,off,off)
  body = []
  for offset in [2**i for i in range(log2(N)+1)]:
    body.append('__implies(%s, %s)' % (lhs(offset), rhs(terms)))
    terms.append('result[left(x,%d)]' % (offset*2))
  return '(' + ' & \\\n  '.join(body) + ')'

def upsweep_core(N):
  return upsweep_pattern(N, lambda terms: 'result[x] == %s' % ' + '.join(terms))

def upsweep_nooverflow(N):
  return upsweep_pattern(N, lambda terms: 'rnooverflow_add_%d(%s)' % (len(terms), ', '.join(terms)))

def upsweep_barrier(N):
  cases = []
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'upsweep(offset,result,len,%s)' % idx
    return '__implies(dlt(tid, %d) & %s(offset, %d), %s)' % (d,rel,offset,rhs)
  ds = [ 2**i for i in reversed(range(log2(N)-1)) ]
  offsets = [ 2**i for i in range(2,log2(N)+1) ]
  assert len(ds) == len(offsets)
  cases.append( gencase(N/2,1,'dgteq','ai') )
  cases += [    gencase(d,offset,'dgteq','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(N/2,2,'dlteq','bi') )
  cases += [    gencase(d,offset,'deq','bi') for d,offset in zip(ds,offsets) ]
  return '(' + ' & \\\n  '.join(cases) + ')'

def upsweep_d_offset(N, include_loop_exit=True):
  offsets = [ 2**i for i in range(log2(N)) ]
  ds = [ x for x in reversed(offsets) ]
  if include_loop_exit:
    offsets += [N]
    ds += [0]
  return '(' + ' | '.join([ '(d == %d & offset == %d)' % (d,offset) for d,offset in zip(ds,offsets) ]) + ')'

def sum_pow2_zeroes(N):
  cases = [ '__ite(dlt(%s, bit) & !isone(%s,x), pow2(%s), %s)' % (i,i,i,0) for i in range(log2(N)-1) ]
  return '(' + ' + \\\n'.join(cases) + ')'

def downsweep_pattern(N,term,identity):
  offsets = [0] + [ 2**i for i in range(log2(N)-1) ]
  xs = range(log2(N))
  cases = [ '__ite(dlteq(offset, %s), %s, %s)' % (offset,term(x),identity) for offset,x in zip(offsets,xs) ]
  return cases

def downsweep_core(N):
  cases = downsweep_pattern(N,(lambda x: 'term(ghostsum,%s,x)' % x),0)
  return '(' + 'result[x] == __ite(isvertex(x,dmul2(offset)), %s, ghostsum[x])' % ' + '.join(cases) + ')'

def downsweep_nooverflow(N):
  cases = downsweep_pattern(N,(lambda x: 'term(ghostsum,%s,x)' % x),0)
  return '(' + '__implies(isvertex(x,dmul2(offset)), rnooverflow_add_%d(%s))' % (log2(N), ', '.join(cases)) + ')'

def downsweep_barrier(N):
  ds = [ 2**i for i in range(log2(N)) ]
  offsets = [x for x in reversed(ds)][1:] + [0]
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'downsweep(offset,result,ghostsum,%s)' % idx
    return '__implies(dlt(tid, %d) & %s(offset, %d), %s)' % (d,rel,offset,rhs)
  cases = [     gencase(d,    offset,    'dgteq','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(ds[0],offsets[0],'dgteq','bi') )
  cases += [    gencase(d,    offset,    'deq'  ,'bi') for d,offset in zip(ds[1:],offsets[1:]) ]
  return '(' + ' & \\\n  '.join(cases) + ')'

def downsweep_d_offset(N,include_loop_exit=True):
  ds = [ 2**i for i in range(log2(N)) ]
  if include_loop_exit:
    ds += [N]
  offsets = [x for x in reversed(ds)]
  return '(' + ' | '.join([ '(d == %s & offset == %s)' % (d,offset) for d,offset in zip(ds,offsets) ]) + ')'

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  if len(argv) != 2:
    print 'error: need [N], number of elements'
    return 1
  N = int(argv[1])
  env = Environment(loader=PackageLoader('genspec', '.'))
  t = env.get_template('spec.template')
  print t.render(N=N, NDIV2=N/2,
    upsweep_core=upsweep_core,
    upsweep_nooverflow=upsweep_nooverflow,
    upsweep_barrier=upsweep_barrier,
    upsweep_d_offset=upsweep_d_offset,
    sum_pow2_zeroes=sum_pow2_zeroes,
    downsweep_core=downsweep_core,
    downsweep_nooverflow=downsweep_nooverflow,
    downsweep_barrier=downsweep_barrier,
    downsweep_d_offset=downsweep_d_offset,
  )

if __name__ == '__main__':
  sys.exit(main())