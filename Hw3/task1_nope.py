import argparse
import json
import time
import pyspark
import findspark
from collections import defaultdict
from itertools import combinations


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc : pyspark.SparkContext):

    review_rdd = sc.textFile(input_file).map(lambda x: json.loads(x))
    review_matrix_rdd = review_rdd.map(lambda x: (x['user_id'],x['business_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x:(x[0],[*set(x[1])]))
    buisness_list = review_rdd.map(lambda x: x['business_id']).distinct().collect()
    bis_len = len(buisness_list)
    hash_list = [lambda x: ((a+3)*hash(x) + int(10000/2) + a) % 10000 for a in range(n_bands * n_rows)]
    sig_matrix = [[float('inf') for _ in range(bis_len)] for _ in range(n_bands * n_rows)]
    minhash_rdd = review_matrix_rdd.map(lambda x: minhash_map(x, buisness_list))
    minhash_rdd.foreach(lambda x: create_signatures(x, sig_matrix, hash_list))
    
    
    print(sig_matrix)
    #minhash_dict = minhash_rdd.collectAsMap()
    # bin_rdd = minhash_rdd.flatMap(lambda x: create_signatures(x, n_bands, n_rows, user_size=len(user_list))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x:(x[0],[*set(x[1])]))
    # canadatePairs_rdd = bin_rdd.flatMap(lambda x: count_bins(x)).distinct()
    # sim = canadatePairs_rdd.map(lambda x: jac_calc(x)).filter(lambda x: x[2] > jac_thr).collect()

    # with open(output_file, 'w') as outfile:
    #     for pair in sim:
    #         json_dict = {'b1':pair[0], 'b2':pair[1], 'sim':pair[2]}
    #         outfile.write(json.dumps(json_dict) + '\n')
 

def minhash_map(line, buisness_list):
    user_id = line[0]
    bis_set = set(line[1])
    min_pos = []
    # minhash
    for i, buisness in enumerate(buisness_list):
        if buisness in bis_set:
            min_pos.append(i)

    return (user_id, min_pos)


def create_signatures(line, sig_matrix, hash_list):
    user_id = line[0]
    min_pos = line[1]
    for i, hash_func in enumerate(hash_list):
        sig_val = hash_func(user_id)
        for pos in min_pos:
            if sig_matrix[i][pos] > sig_val:
                sig_matrix[i][pos] = sig_val
    return


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
    s1_len = len(s1)
    s2_len = len(s2)
    if s1_len > s2_len:
        min_set = s2
        max_set = set(s1)
    else:
        min_set = s1
        max_set = set(s2)
    sim_cout = 0
    for position in min_set:
        if position in max_set:
            sim_cout += 1
    return sim_cout / max(s1_len,s2_len)


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


