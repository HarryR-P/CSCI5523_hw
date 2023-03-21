import pyspark
import argparse
import json
import gc
import time
from itertools import chain, combinations
from collections import defaultdict
#import findspark

def main(args):
    start_time = time.time()

    #findspark.init()

    sc_conf = pyspark.SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executer.memory', '4g')
    
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    input_path = args.input_file
    output_path = args.output_file
    k = args.k
    support_thr = args.s

    input_rdd = sc.textFile(input_path).map(lambda x: x.split(','))
    # header = input_rdd.first()
    # input_rdd = input_rdd.filter(lambda x: x != header)

    case_rdd = input_rdd.map(lambda x: (str(x[0]), str(x[1]))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    case_rdd = case_rdd.filter(lambda x: len(x[1]) > k).map(lambda x: (x[0],sorted([*set(x[1])])))

    num_partitions = case_rdd.getNumPartitions()
    canidates_rdd = case_rdd.mapPartitions(lambda x: pass1_map(x, support_thr, num_partitions)).reduceByKey(lambda a,b: a)
    canidates = canidates_rdd.map(lambda x: x[0]).sortBy(lambda x: (len(x),x)).collect()

    frequent_rdd = case_rdd.mapPartitions(lambda x: pass2_map(x,  canidates)).reduceByKey(lambda a,b: a+b).filter(lambda x: x[1] >= support_thr)
    frequent = frequent_rdd.map(lambda x: x[0]).sortBy(lambda x: (len(x),x)).collect()

    return_dict = {"Candidates": canidates, "Frequent Itemsets": frequent, "Runtime": time.time() - start_time}

    with open(output_path, 'w') as outfile:
        json.dump(return_dict, outfile)
    
    sc.stop()
    return

def pass1_map(baskets, support_thr, n):
    single_count = defaultdict(int)
    multi_count = defaultdict(int)
    partition_thr = support_thr / n
    baskets_list = list(baskets)

    for _ , basket in baskets_list:   
        for item in basket:
            single_count[item] += 1

    for _ , basket in baskets_list:
        subset_iter = subsets(basket)
        for subset in subset_iter:
            freq_flag = []
            for item in subset:
                freq_flag.append(single_count[item] >= partition_thr)
            if all(freq_flag):
                multi_count[subset] += 1
    
    candidates = []
    for key in single_count.keys():
        if single_count[key] >= partition_thr:
            candidates.append(((key,), 1))
    for key in multi_count.keys():
        if multi_count[key] >= partition_thr:
            candidates.append((key, 1))

    return iter(candidates)


def pass2_map(baskets, canidates):
    count = defaultdict(int)
    canidates = set(canidates)
    for _, basket in baskets:
        for item in basket:
            if (item,) in canidates:
                count[(item,)] += 1

        subset_iter = subsets(basket)

        for subset in subset_iter:
            if subset in canidates:
                count[subset] += 1

    return_list = []
    for key in count.keys():
        return_list.append((key, count[key]))

    return iter(return_list)


def subsets(basket):
    return chain.from_iterable(combinations(basket, r) for r in range(2, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2T1')
    parser.add_argument('--input_file', type=str, default='./data/hw2/gen.csv', help='the input file')
    parser.add_argument('--output_file', type=str, default='./data/hw2/a2t2.json',
                        help='the output file that contains your answers')
    parser.add_argument('--k', type=int, default=10, help='Filter qualified users')
    parser.add_argument('--s', type=int, default=10, help='Support threshold')

    args = parser.parse_args()
    main(args)