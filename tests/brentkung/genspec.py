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
  body.append(read_permission('len[x]'))
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

def foldright(f,xs,i):
  return reduce(lambda x,y: f(y,x), reversed(xs),i)

def ilog2(N):
  def gencase(i):
    return '__ite_dtype((%d <= x) & (x < %d), %s' % (2**i, 2**(i+1), i)
  cases = [ '__ite_dtype(x == 1, 0' ]
  cases.extend([ gencase(i) for i in range(1,log2(N)-1) ])
  return foldright(lambda x,y: '(%s, %s))' % (x,y), cases, '%d' % (log2(N)-1))

def downsweep_term(N):
  def gencase(i):
    return '__ite_dtype((%d <= i) & isone(%d, (x+1)), %d, 0)' % (i,i,2**i)
  cases = [ gencase(i) for i in range(log2(N)-1) ]
  return '(x - (%s))' % ' + '.join(cases)

def downsweep_terms_pattern(N,term,identity):
  def gencase(i):
    return '__ite_rtype(isone(%d,(x+1)) & (%d < ilog2(x+1)), %s, %s)' % (i,i,term(i),identity)
  return [ gencase(i) for i in reversed(range(log2(N)-1)) ]

def downsweep_summation(N):
  def term(x): return 'ghostsum[term(x,%s)]' % x
  cases = downsweep_terms_pattern(N,term,0)
  cases.append('ghostsum[x]')
  return '(result[x] == (%s))' % summation(cases, 'raddf')

def downsweep_updated_condition(N):
  def gencase(i):
    u,v = 2**i, 2**(i-1)
    return '((offset <= %d) & updated(x,%d))' % (u,v)
  return ' | '.join([ gencase(i) for i in reversed(range(1,log2(N))) ])

def downsweep_core(N):
  # case where result[x] has not being updated
  offsets = [ '(offset == %s)' % 2**i for i in reversed(range(1,log2(N)+1)) ]
  updated = [ '!updated(x,%s)' % 2**i for i in reversed(range(log2(N)-1)) ]
  terms = offsets[:1] + [ '%s & (%s' % (u,o) for u,o in zip(updated,offsets[1:]) ]
  nbrackets = len(offsets) + len(updated)
  lhs = '(' + reduce(lambda x,y: '%s | \\\n            (%s' % (x,y), terms) + ')' * nbrackets
  cases = [ '__implies(%s, (result[x] == ghostsum[x]))' % lhs ]
  # result[x] has been updated
  cases.append( '__implies(%s, downsweep_summation(result,ghostsum,x))' % downsweep_updated_condition(N))
  return '(' + ' & \\\n   '.join(cases) + ')'

def downsweep_nooverflow(N):
  def term(x): return 'ghostsum[term(x,%s)]' % x
  cases = downsweep_terms_pattern(N,term,0)
  cases.append('ghostsum[x]')
  rhs = '__add_noovfl(%s)' % ', '.join(cases)
  return '__implies(%s, %s)' % (downsweep_updated_condition(N), rhs)

def downsweep_barrier(N,mul2shift=False):
  def genrhs(aibi,f,offset):
    idx = '%s_idx(%d,tid)' % (aibi,f(offset))
    return 'downsweep(offset,result,ghostsum,%s)' % idx
  def gencase(d,offset,rel,aibi,f=lambda offset:offset/2):
    lhs = '(tid < %s) & (offset %s %d)' % (d,rel,offset)
    rhs = genrhs(aibi,f,offset)
    return '__implies(%s, %s)' % (lhs,rhs)
  ds = [ 2**i for i in range(1,log2(N)) ]
  offsets = [ x for x in reversed(ds) ]
  cases =      [ gencase(d,offset,'>','ai')                  for d,offset in reversed(zip(ds,offsets)) ]
  cases.append('__implies((tid < 1),                downsweep(offset,result,ghostsum,ai_idx(div2(offset),tid)))')
  cases.append('__implies((tid < 1),                %s)' % genrhs('bi',lambda offset:offset/2,N))
  cases.extend([ gencase(d-1,offset,'==','lf_ai',lambda x:x) for d,offset in          zip(ds,offsets)  ])
  cases.extend([ gencase(d-1,offset,'==','lf_bi',lambda x:x) for d,offset in          zip(ds,offsets)  ])
  return '(' + ' & \\\n   '.join(cases) + ')'

def downsweep_d_offset(N,include_loop_exit=True):
  ds = [ 2**i for i in range(1,log2(N)) ]
  if include_loop_exit: ds.append(N)
  offsets = [ x for x in reversed(ds) ]
  return '(' + ' | '.join([ '((d == %d) & (offset == %d))' % (d,offset) for d,offset in zip(ds,offsets) ]) + ')'

def upsweep_instantiation(N,elementwise=False):
  cases = [ 'tid', '(tid+1)' ]
  if elementwise: cases = [ 'x2t(tid)', '(x2t(tid)+1)' ]
  for x in range(1,log2(N)-1):
    term = 'div2(%s)' % cases[-1]
    cases.append(term)
  cases.extend([ x.replace('tid', 'other_tid') for x in cases ])
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
  return '(' + ' & \\\n   '.join(cases) + ')'

def final_downsweep_barrier(N,mul2shift=False):
  cases = []
  cases.append('__implies((tid < 1), downsweep(/*offset=*/2,result,ghostsum,ai_idx(1,tid)))')
  cases.append('__implies((tid < 1), downsweep(/*offset=*/2,result,ghostsum,bi_idx(%d,tid)))' % (N/2))
  cases.append('__implies((tid < %d), downsweep(/*offset=*/2,result,ghostsum,lf_ai_idx(2,tid)))' % (N/2-1))
  cases.append('__implies((tid < %d), downsweep(/*offset=*/2,result,ghostsum,lf_bi_idx(2,tid)))' % (N/2-1))
  return '(' + ' & \\\n   '.join(cases) + ')'

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
    ilog2=ilog2,
    downsweep_term=downsweep_term,
    downsweep_summation=downsweep_summation,
    downsweep_core=downsweep_core,
    downsweep_nooverflow=downsweep_nooverflow,
    downsweep_barrier=downsweep_barrier,
    downsweep_d_offset=downsweep_d_offset,
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
