import argparse
import json
import time
import pyspark
import findspark
from random import shuffle
import task1
from collections import defaultdict
from itertools import permutations

def main():
    l = [1,2]
    c = list(permutations(l,2))
    print(c)
    return


if __name__ == '__main__':
    main()