import argparse
import json
import time
import pyspark
from itertools import combinations


def main(train_file, model_file, co_rated_thr, sc : pyspark.SparkContext):

    review_rdd = sc.textFile(train_file).map(lambda x: json.loads(x))
    user_review_matrix_rdd = review_rdd.map(lambda x: (x['user_id'],x['business_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)\
                                .map(lambda x: (x[0], [*set(x[1])]))
    co_rated_rdd = user_review_matrix_rdd.flatMap(lambda x: map_co_rated(x)).reduceByKey(lambda a,b:a+b).filter(lambda x: x >= co_rated_thr)
    bis_review_matrix_rdd = review_rdd.map(lambda x: (x['business_id'],(x['user_id'], x['stars']))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    bis_review_matrix_rdd = bis_review_matrix_rdd
    user_set = review_rdd.map(lambda x: x['user_id']).distinct().collect()
    sig_matrix_rdd = review_matrix_rdd.map(lambda x: map_to_matrix(x, bisness_set))
    


def map_to_matrix(line, bisness_set):
    user_id = line[0]
    ratings_list = line[1]
    rating_output = []
    seen = []
    for rating in ratings_list:
        if rating[0] not in seen:
            rating_output.append(rating)
            seen.append(rating[0])
    return (user_id, rating_output)


def map_co_rated(line):
    ratings_list = line[1]
    if len(ratings_list) < 2: return []
    pairs = combinations(ratings_list,2)
    return [(tuple(sorted(pair, key=lambda x: x[0])),1) for pair in pairs]



if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--time_file', type=str, default='./outputs/time.out')
    parser.add_argument('--m', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.m, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))
