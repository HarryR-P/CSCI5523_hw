import argparse
import json
import time
import pyspark
#import findspark
import math
from itertools import combinations


def main(train_file, model_file, co_rated_thr, sc : pyspark.SparkContext):

    review_rdd = sc.textFile(train_file).map(lambda x: json.loads(x))
    user_review_matrix_rdd = review_rdd.map(lambda x: (x['user_id'],x['business_id'])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)\
                                .map(lambda x: (x[0], [*set(x[1])]))
    co_rated_rdd = user_review_matrix_rdd.flatMap(lambda x: map_co_rated(x)).reduceByKey(lambda a,b:a+b).filter(lambda x: x[1] >= co_rated_thr).map(lambda x: (x[0][0],x[0][1]))
    bis_review_matrix_rdd = review_rdd.map(lambda x: (x['business_id'],(x['user_id'], x['stars']))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    sig_dict_rdd = bis_review_matrix_rdd.map(lambda x: map_to_matrix(x))
    pair_rdd = co_rated_rdd.join(sig_dict_rdd).map(lambda x: (x[1][0],(x[0],x[1][1]))).join(sig_dict_rdd).map(lambda x: ((x[1][0][0],x[1][0][1]),(x[0],x[1][1])))
    model = pair_rdd.map(lambda x: calc_corr(x)).collect()
    
    with open(model_file, 'w') as outfile:
        for pair in model:
            json_dict = {'b1':pair[0], 'b2':pair[1], 'sim':round(pair[2],2)}
            outfile.write(json.dumps(json_dict) + '\n')
    

def map_to_matrix(line):
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
    return [(tuple(sorted(pair)),1) for pair in pairs]


def calc_corr(line):
    b1 = line[0][0]
    b2 = line[1][0]
    sig_dict_1 = {uid: stars for uid, stars in line[0][1]}
    sig_dict_2 = {uid: stars for uid, stars in line[1][1]}
    avg1 = sum(sig_dict_1.values()) / len(sig_dict_1)
    avg2 = sum(sig_dict_2.values()) / len(sig_dict_2)
    
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for uid in sig_dict_1:
        if uid in sig_dict_2:
            numerator += (sig_dict_1[uid] - avg1)*(sig_dict_2[uid] - avg2)
            denominator1 += (sig_dict_1[uid] - avg1)**2
            denominator2 += (sig_dict_2[uid] - avg2)**2
    if denominator1 != 0.0 and denominator2 != 0.0:
        corr = numerator / (math.sqrt(denominator1)*math.sqrt(denominator2))
    else:
        corr = 0.0
    return (b1,b2,corr)


if __name__ == '__main__': 
    #findspark.init()
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
