import pyspark
import argparse
import re
import json
import findspark
import itertools
from itertools import chain, combinations
from collections import defaultdict

def main():
    d = defaultdict(int)
    d['a'] += 1
    print(d)
    return


def subsets(basket):
    twos = list(itertools.combinations(basket, 2))
    threes = list(itertools.combinations(basket, 3))
    fours = list(itertools.combinations(basket, 4))
    return twos + threes + fours


if __name__ == '__main__':
    main()