#!/usr/bin/env python

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

def gengbpl(width,N):
  if width < 32:
    print 'error: [width] must be greater-than-or-equal to 32'
    return 1
  if not ispow2(width):
    print 'error: [width] must be a power of two'
    return 1
  if not ispow2(N):
    print 'error: [N] must be a power of two'
    return 1
  env = Environment(loader=PackageLoader('gengbpl', '.'))
  t = env.get_template('template.gbpl')
  return t.render(nthreads='%dbv32' % N, abstract_type = 'bv%d' % width, width=width)

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  if len(argv) != 3:
    print 'error: need [width], bitwidth and [N], number of elements'
    return 1
  width = int(argv[1])
  N = int(argv[2])
  print gengbpl(width,N)

if __name__ == '__main__':
  sys.exit(main())
