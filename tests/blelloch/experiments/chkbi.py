#!/usr/bin/env python

from build import CHECK, build_and_run, main
import sys

header = 'Only BI-checking'

def doit():
  checks = [CHECK.BI]
  extraflags = ['--no-barrier-access-checks','--only-divergence','--asymmetric-asserts']
  build_and_run(checks,extraflags)

if __name__ == '__main__':
  sys.exit(main(doit,header))
