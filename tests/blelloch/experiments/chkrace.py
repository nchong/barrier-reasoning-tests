#!/usr/bin/env python

from build import CHECK, build_and_run, main
import sys

header = 'Only race-checking'

def doit():
  build_and_run([CHECK.RACE])

if __name__ == '__main__':
  sys.exit(main(doit,header))
