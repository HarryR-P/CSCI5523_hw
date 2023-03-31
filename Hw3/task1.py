import argparse
import json
import time
import pyspark
import findspark
from collections import defaultdict
from itertools import combinations


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc : pyspark.SparkContext):

    review_rdd = sc.textFile(input_file).map(lambda x: json.loads(x))
    review_matrix_rdd = review_rdd.map(lambda x: (x['business_id'],x['user_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x:(x[0],[*set(x[1])]))
    user_list = review_rdd.map(lambda x: x['user_id']).distinct().collect()
    minhash_rdd = review_matrix_rdd.map(lambda x: minhash_map(x, user_list))
    #minhash_dict = minhash_rdd.collectAsMap()
    bin_rdd = minhash_rdd.flatMap(lambda x: create_signatures(x, user_list, n_bands, n_rows, user_size=len(user_list))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x:(x[0],[*set(x[1])]))
    canadatePairs_rdd = bin_rdd.flatMap(lambda x: count_bins(x)).distinct()
    sim = canadatePairs_rdd.map(lambda x: jac_calc(x)).filter(lambda x: x[2] > jac_thr).collect()

    with open(output_file, 'w') as outfile:
        for pair in sim:
            json_dict = {'b1':pair[0], 'b2':pair[1], 'sim':pair[2]}
            outfile.write(json.dumps(json_dict) + '\n')
 

def minhash_map(line, user_list):
    business_id = line[0]
    ratings_set = line[1]
    position_list = []
    # minhash
    for rating_id in ratings_set:
        idx = user_list.index(rating_id)
        position_list.append(idx)
    return (business_id, tuple(position_list))


def create_signatures(line, user_list, n_bands, n_rows, user_size):
    business_id = line[0]
    min_hash = line[1]
    # create signature
    signature_buckets = n_bands * n_rows
    min_sig = [float('inf') for _ in range(signature_buckets)]
    hash_function = lambda x,a: ((a+3)*hash(x) + int(user_size/2) + a) % user_size
    for i in range(signature_buckets):
        for position in min_hash:
            index = hash_function(user_list[position], i)
            if index < min_sig[i]:
                min_sig[i] = index
    # seperate bands
    # band_list = [min_sig[i:i+n_rows] for i in range(0, signature_buckets, n_rows)]
    # hash-to-bins
    # hash_func = lambda band: (sum([359*val for val in band]) + int((buckets/2))) % buckets
    # bin_list = [hash_func(band) for band in band_list]
    return [(tuple(min_sig[i:i+n_rows]), (business_id,min_hash)) for i in range(0, signature_buckets, n_rows)]


def count_bins(line):
    business_id = line[1]
    if len(business_id) < 2: return []
    pairs = combinations(range(len(business_id)), 2)
    return [tuple(sorted((business_id[pair[0]], business_id[pair[1]]), key=lambda x: x[0])) for pair in pairs]


def jac_calc(line):
    id_1 = line[0][0]
    id_2 = line[1][0]
    sig_1 = line[0][1]
    sig_2 = line[1][1]
    sim = jacobian(sig_1, sig_2)
    return (id_1, id_2, sim)


def jacobian(s1, s2):
    s2_set = set(s2)
    sim_cout = 0
    for position in s1:
        if position in s2_set:
            sim_cout += 1
    return sim_cout / max(len(s1),len(s2))


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


