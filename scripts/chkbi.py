#!/usr/bin/env python

from build import CHECK, build_and_run, main
import sys

header = 'Only BI-checking'

def doit():
  checks = [CHECK.BI]
  extraflags = ['--no-barrier-access-checks','--only-divergence','--asymmetric-asserts']
  return build_and_run(checks,extraflags)

def main_wrapper(argv):
  main(doit,header,argv)

if __name__ == '__main__':
  sys.exit(main(doit,header))
