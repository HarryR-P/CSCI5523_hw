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
    l = [('a',1),('b',2), ('c',6), ('d',5), ('e',4)]
    l.sort(key=lambda x: x[1], reverse=True)
    h = l[:2]
    print(h)
    return


if __name__ == '__main__':
    main()