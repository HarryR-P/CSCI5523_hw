import argparse
import json
import time
import pyspark
import findspark
from itertools import combinations


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc : pyspark.SparkContext):

    review_rdd = sc.textFile(input_file).map(lambda x: json.loads(x))
    review_matrix_rdd = review_rdd.map(lambda x: (x['business_id'],x['user_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)\
        .map(lambda x: (x[0],sorted([*set(x[1])])))
    user_list = review_rdd.map(lambda x: x['user_id']).distinct().sortBy(lambda x: x).collect()
    sig_rdd = review_matrix_rdd.map(lambda x: minhash_map(x, user_list))
    sig_dict = sig_rdd.collectAsMap()
    bin_rdd = sig_rdd.flatMap(lambda x: seperate_bands(x, band_num=n_bands, band_bins=n_rows)).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    canadatePairs_rdd = bin_rdd.flatMap(lambda x: count_bins(x)).distinct()

    

def minhash_map(line, user_list):
    business_id = line[0]
    ratings_set = set(line[1])
    bins = len(user_list)
    hash_function = lambda x,a: (a*x + 25) % bins
    bit_list = []
    # minhash
    for user_id in user_list:
        if user_id in ratings_set:
            bit_list.append(1)
        else:
            bit_list.append(0)
    # signature
    signature_buckets = 30
    min_sig = [float('inf') for _ in range(signature_buckets)]
    for a in range(signature_buckets):
        for position  in range(len(user_list)):
            index = hash_function(position, a)
            if bit_list[position] == 1 and min_sig[a] > index:
                min_sig[a] = index
    return (business_id, min_sig)


def seperate_bands(line, band_num, band_bins):
    business_id = line[0]
    sig = line[1]
    row_size = int(len(sig) / band_num)
    band_list = [sig[i:i+row_size] for i in range(0, len(sig), row_size)]
    # hash-to-bins
    hash_func = lambda band: (5*sum(band) + 10) % band_bins
    bin_list = [hash_func(band) for band in band_list]
    return [(bin, business_id) for bin in bin_list]


def count_bins(line):
    business_id = set(line[1])
    bin = line[0]
    if len(business_id) < 2: return []
    pairs = combinations(business_id, 2)
    return [tuple(sorted(pair)) for pair in pairs]


def jac_calc(line, sig_dict):
    id_1 = line[0]
    id_2 = line[1]
    sig_1 = sig_dict[id_1]
    sig_2 = sig_dict[id_2]
    sim = jacobian(sig_1, sig_2)
    return ((id_1, id_2),sim)



def jacobian(s1, s2):
    sim_cout = 0
    for el1, el2 in zip(s1, s2):
        if el1 == el2:
            sim_cout += 1
    return sim_cout / len(s1)


if __name__ == '__main__':
    findspark.init()
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_file', type=str, default='./data/train_review.json')
    parser.add_argument('--output_file', type=str, default='./outputs/task1.out')
    parser.add_argument('--time_file', type=str, default='./outputs/task1.time')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--n_bands', type=int, default=50)
    parser.add_argument('--n_rows', type=int, default=2)
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold, args.n_bands, args.n_rows, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))


