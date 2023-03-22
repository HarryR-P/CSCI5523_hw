import argparse
import json
import time
import pyspark
import findspark


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc):

    review_rdd = sc.textFile(input_file).map(lambda x: json.loads(x))
    review_matrix_rdd = review_rdd.map(lambda x: (x['business_id'],x['user_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)\
        .map(lambda x: (x[0],sorted([*set(x[1])])))
    user_list = review_rdd.map(lambda x: x['user_id']).distinct().sortBy(lambda x: x).collect()
    minhash_rdd = review_matrix_rdd.map(lambda x: minhash_map(x, user_list))
    print(minhash_rdd.take(5))


def minhash_map(line, user_list):
    business_id = line[0]
    ratings_list = set(line[1])
    bins = len(user_list)
    hash_function = lambda x, a: (a*hash(x) + 25) % bins
    bit_list = []
    for user_id in user_list:
        if user_id in ratings_list:
            bit_list.append(1)
        else:
            bit_list.append(0)
    signature_buckets = 25
    min_sig = [0 for _ in range(signature_buckets)]
    for a in range(signature_buckets):
        for i  in range(len(user_list)):
            perm_index = hash_function(i, a)
            if bit_list[perm_index]:
                min_sig[a] = perm_index
                break
    
    return (business_id, min_sig)


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


