#!/usr/bin/env python

from build import CHECK, build_and_run, main
import sys

header = 'All checks (race, BI and BI access)'

def doit():
  checks = [CHECK.RACE, CHECK.BI_ACCESS, CHECK.BI]
  return build_and_run(checks)

def main_wrapper(argv):
  main(doit,header,argv)

if __name__ == '__main__':
  sys.exit(main(doit,header))
