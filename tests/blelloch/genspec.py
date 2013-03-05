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

def ispow2(x):
  return x != 0 and ((x & (x-1)) == 0)

def log2(x):
  return int(log(x,2))

def summation(terms,f):
  return reduce(lambda x,y: '%s(%s,%s)' % (f,x,y), terms)

def read_permission(x):
  return '__read_permission(%s)' % x

def upsweep_pattern(N,rhs):
  terms = [ 'len[x]' ]
  def lhs(off):
    return '(((offset == %d) & isvertex(x,offset)) | ((%d < offset) & stopped(x,%d)))' % (off,off,off)
  body = []
  for offset in [2**i for i in range(log2(N)+1)]:
    body.append('__implies(%s, %s)' % (lhs(offset), rhs(terms)))
    terms.append('result[left(x,%d)]' % (offset*2))
  return '(' + ' & \\\n  '.join(body) + ')'

def upsweep_core(N):
  return upsweep_pattern(N, lambda terms: 'result[x] == %s' % summation(reversed(terms), 'raddf'))

def upsweep_nooverflow(N):
  return upsweep_pattern(N, lambda terms: '__add_noovfl(%s)' % ', '.join(terms))

def upsweep_barrier(N):
  cases = []
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'upsweep(offset,result,len,%s)' % idx
    return '__implies((tid < %d) & (offset %s %d), %s)' % (d,rel,offset,rhs)
  ds = [ 2**i for i in reversed(range(log2(N)-1)) ]
  offsets = [ 2**i for i in range(2,log2(N)+1) ]
  assert len(ds) == len(offsets)
  cases.append( gencase(N/2,1,'>=','ai') )
  cases += [    gencase(d,offset,'>=','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(N/2,2,'<=','bi') )
  cases += [    gencase(d,offset,'==','bi') for d,offset in zip(ds,offsets) ]
  return '(' + ' & \\\n  '.join(cases) + ')'

def upsweep_d_offset(N, include_loop_exit=True):
  offsets = [ 2**i for i in range(log2(N)) ]
  ds = [ x for x in reversed(offsets) ]
  if include_loop_exit:
    offsets += [N]
    ds += [0]
  return '(' + ' | '.join([ '(d == %d & offset == %d)' % (d,offset) for d,offset in zip(ds,offsets) ]) + ')'

def ghostreadvars(N):
  offsets = [2**i for i in range(1, log2(N)+1)]
  return [ 'ghostread%d' % offset for offset in offsets ]

def upsweep_permissions(N):
  def lhs(off):
    return '(((offset == %d) && isvertex(x,offset)) || ((%d < offset) && stopped(x,%d)))' % (off,off,off)
  def rhs(terms):
    return '%s = true;' % ' = '.join(terms)
  body = [ '%s = false;' % ' = '.join(ghostreadvars(N)) ]
  terms = [ 'ghostread2' ]
  offsets = [2**i for i in range(1, log2(N)+1)]
  for offset in offsets:
    body.append('if (%s) %s' % (lhs(offset), rhs(terms)))
    terms.append('ghostread%d' % (offset*2))
  body.append(read_permission('result[x]'))
  for offset in offsets:
    body.append('if (ghostread%s) %s' % (offset, read_permission('result[left(x,%d)]' % offset)))
  return '{' + ' \\\n  '.join(body) + '}'

def upsweep_barrier_permissions(N):
  cases = []
  cases.append('bool %s;' % ', '.join(ghostreadvars(N)))
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'upsweep_permissions(offset,result,len,%s)' % idx
    return 'if ((tid < %d) && (offset %s %d)) %s' % (d,rel,offset,rhs)
  ds = [ 2**i for i in reversed(range(log2(N)-1)) ]
  offsets = [ 2**i for i in range(2,log2(N)+1) ]
  assert len(ds) == len(offsets)
  cases.append( gencase(N/2,1,'>=','ai') )
  cases += [    gencase(d,offset,'>=','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(N/2,2,'<=','bi') )
  cases += [    gencase(d,offset,'==','bi') for d,offset in zip(ds,offsets) ]
  return '{' + ' \\\n  '.join(cases) + '}'

def sum_pow2_zeroes(N):
  cases = [ '__ite_dtype((%s < bit) & !isone(%s,x), pow2(%s), %s)' % (i,i,i,0) for i in range(log2(N)-1) ]
  return '(' + ' + \\\n'.join(cases) + ')'

def downsweep_pattern(N,term,identity):
  offsets = [0] + [ 2**i for i in range(log2(N)-1) ]
  xs = range(log2(N))
  cases = [ '__ite_rtype((offset <= %s), %s, %s)' % (offset,term(x),identity) for offset,x in reversed(zip(offsets,xs)) ]
  return cases

def downsweep_core(N):
  cases = downsweep_pattern(N,(lambda x: 'term(ghostsum,%s,x)' % x),'ridentity')
  return '(' + 'result[x] == __ite_rtype(isvertex(x,mul2(offset)), %s, ghostsum[x])' % summation(cases, 'raddf') + ')'

def downsweep_nooverflow(N):
  cases = downsweep_pattern(N,(lambda x: 'term(ghostsum,%s,x)' % x),'ridentity')
  return '(' + '__implies(isvertex(x,mul2(offset)), __add_noovfl(%s))' % ', '.join(cases) + ')'

def downsweep_barrier(N):
  ds = [ 2**i for i in range(log2(N)) ]
  offsets = [x for x in reversed(ds)][1:] + [0]
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'downsweep(offset,result,ghostsum,%s)' % idx
    return '__implies((tid < %d) & (offset %s %d), %s)' % (d,rel,offset,rhs)
  cases = [     gencase(d,    offset,    '>=','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(ds[0],offsets[0],'>=','bi') )
  cases += [    gencase(d,    offset,    '=='  ,'bi') for d,offset in zip(ds[1:],offsets[1:]) ]
  return '(' + ' & \\\n  '.join(cases) + ')'

def downsweep_d_offset(N,include_loop_exit=True):
  ds = [ 2**i for i in range(log2(N)) ]
  if include_loop_exit:
    ds += [N]
  offsets = [x for x in reversed(ds)]
  return '(' + ' | '.join([ '(d == %s & offset == %s)' % (d,offset) for d,offset in zip(ds,offsets) ]) + ')'

def downsweep_permissions(N):
  terms = [ 'result[x]' ]
  for x in range(log2(N)):
    terms.append('ghostsum[x + sum_pow2_zeroes(%d,x) - pow2(%d)]' % (x,x))
  body = [ read_permission(x) for x in terms ]
  return '{' + ' \\\n  '.join(body) + '}'

def downsweep_barrier_permissions(N):
  ds = [ 2**i for i in range(log2(N)) ]
  offsets = [x for x in reversed(ds)][1:] + [0]
  def gencase(d,offset,rel,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'downsweep_permissions(offset,result,ghostsum,%s)' % idx
    return 'if ((tid < %d) && (offset %s %d)) %s' % (d,rel,offset,rhs)
  cases = [     gencase(d,    offset,    '>=','ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(ds[0],offsets[0],'>=','bi') )
  cases += [    gencase(d,    offset,    '=='  ,'bi') for d,offset in zip(ds[1:],offsets[1:]) ]
  return '{' + ' \\\n  '.join(cases) + '}'

def upsweep_instantiation(N,elementwise=False):
  cases = [ 'tid' ]
  for x in range(1,log2(N)-1):
    term = 'div2(%s)' % cases[-1]
    cases.append(term)
  cases.append('other_tid')
  if elementwise: cases = [ 'x2t(%s)' % x for x in cases ]
  return ', '.join(cases)

def final_upsweep_barrier(N):
  cases = []
  def gencase(d,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'upsweep(/*offset=*/N,result,len,%s)' % idx
    return '__implies((tid < %d), %s)' % (d,rhs)
  ds = [ 2**i for i in reversed(range(log2(N)-1)) ]
  offsets = [ 2**i for i in range(2,log2(N)+1) ]
  assert len(ds) == len(offsets)
  cases.append( gencase(N/2,'ai') )
  cases += [    gencase(d,'ai') for d,offset in zip(ds,offsets) ]
  cases.append( gencase(1,'bi') )
  return '(' + ' & \\\n  '.join(cases) + ')'

def final_downsweep_barrier(N):
  cases = []
  def gencase(d,aibi):
    idx = '%s_idx(%d,tid)' % (aibi,N/2/d)
    rhs = 'downsweep(/*offset=*/0,result,ghostsum,%s)' % idx
    return '__implies((tid < %d), %s)' % (d,rhs)
  cases.append( gencase(N/2,'ai') )
  cases.append( gencase(N/2,'bi') )
  return '(' + ' & \\\n  '.join(cases) + ')'

def genspec(N):
  if not ispow2(N):
    print 'error: [N] must be a power of two'
    return 1
  env = Environment(loader=PackageLoader('genspec', '.'))
  t = env.get_template('spec.template')
  return t.render(N=N, NDIV2=N/2,
    upsweep_core=upsweep_core,
    upsweep_nooverflow=upsweep_nooverflow,
    upsweep_barrier=upsweep_barrier,
    upsweep_d_offset=upsweep_d_offset,
    upsweep_permissions=upsweep_permissions,
    upsweep_barrier_permissions=upsweep_barrier_permissions,
    sum_pow2_zeroes=sum_pow2_zeroes,
    downsweep_core=downsweep_core,
    downsweep_nooverflow=downsweep_nooverflow,
    downsweep_barrier=downsweep_barrier,
    downsweep_d_offset=downsweep_d_offset,
    downsweep_permissions=downsweep_permissions,
    downsweep_barrier_permissions=downsweep_barrier_permissions,
    upsweep_instantiation=upsweep_instantiation,
    final_upsweep_barrier=final_upsweep_barrier,
    final_downsweep_barrier=final_downsweep_barrier,
  )

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  if len(argv) != 2:
    print 'error: need [N], number of elements'
    return 1
  N = int(argv[1])
  print genspec(N)

if __name__ == '__main__':
  sys.exit(main())
