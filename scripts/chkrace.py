#!/usr/bin/env python

from build import CHECK, build_and_run, main
import sys

header = 'Only race-checking'

def doit():
  return build_and_run([CHECK.RACE])

def main_wrapper(argv):
  main(doit,header,argv)

if __name__ == '__main__':
  sys.exit(main(doit,header))
