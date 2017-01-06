#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read a PHYLIP-format file and produce an appropriate config file for passing to `dnapars`.

`dnapars` is a rather old program that doesn't play very well in a
pipeline.  It prompts the user for configuration information and reads
responses from stdin.  The config file generated by this script is
meant to mimic the responses to the expected prompts.

Typical usage is,

     $ mkdnamlconfig.py sequence.phy --germline GLid > dnapars.cfg
     $ dnapars < dnapars.cfg
"""
import re
import os
import argparse
from warnings import warn

def extract_germline(file, germline):
    with open(file, 'r') as fh:
        for lineno, line in enumerate(fh):
            if re.match(germline, line):
                return lineno

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phylip', help='PHYLIP input', type=str)
    parser.add_argument('--germline', default='GL', help='germline sequence id', type=str)
    args = parser.parse_args()

    germline_idx = extract_germline(args.phylip, args.germline)
    print(args.phylip)	    # phylip input file
    print('O')		        # Outgroup root
    print(germline_idx)	    # naive index in phylip
    print('J')              # jumble
    print('13')
    print('10')
    print('4')
    print('5')
    print('.')
    print('Y')

if __name__ == "__main__":
   main()