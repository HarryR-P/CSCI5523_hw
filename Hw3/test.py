import argparse
import json
import time
import pyspark
import findspark
from collections import defaultdict
from itertools import combinations

def main():
    a = (1,2,3,4)
    for i in a:
        print(i)
    return


def gen_primes():
    seen = {}
    cur = 2
    while True:
        if cur not in seen:
            yield cur
            seen[cur * cur] = [cur]
        else:
            for prime in seen[cur]:
                seen.setdefault(prime + cur, []).append(prime)
            del seen[cur]
        cur += 1


if __name__ == '__main__':
    main()