import argparse
import json
import time
import pyspark
import findspark
from collections import defaultdict
from itertools import combinations

def main():
    start = time.time()
    user_list = list(range(50000))

    ratings_set = [1, 6250, 7000, 40000]
    bins = len(user_list)
    hash_function = lambda x,a: ((a + 1)*x + 1000) % bins
    minHash_dict = defaultdict(bool)
    bit_list = []
    # minhash
    for rating_id in ratings_set:
        idx = user_list.index(rating_id)
        minHash_dict[idx] = True
    print(time.time() - start)
    signature_buckets = 100
    min_sig = []
    bucket_list = list(range(signature_buckets))
    for position in range(len(user_list)):
        indexs = [hash_function(position, a) for a in range(signature_buckets)]
        for index in indexs:
            if minHash_dict[index]:
                min_sig.append(index)

    print(time.time() - start)
    return

def jacobian(s1, s2):
    sim_cout = 0
    for el1, el2 in zip(s1, s2):
        if el1 == el2:
            sim_cout += 1
    return sim_cout / len(s1)

if __name__ == '__main__':
    main()