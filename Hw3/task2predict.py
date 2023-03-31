import argparse
import json
import time
import pyspark
import math
#import findspark



def main(train_file, test_file, model_file, output_file, n_weights, sc : pyspark.SparkContext):
    review_rdd = sc.textFile(train_file).map(lambda x: json.loads(x))
    test_rdd = sc.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],x['business_id']))
    model_rdd = sc.textFile(model_file).map(lambda x: json.loads(x)).flatMap(lambda x: [(x['b1'],(x['b2'],x['sim'])),(x['b2'],(x['b1'],x['sim']))]).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    bis_review_matrix_rdd = review_rdd.map(lambda x: (x['user_id'],(x['business_id'], x['stars']))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x: map_to_matrix(x))
    user_sig_rdd = test_rdd.join(bis_review_matrix_rdd).map(lambda x: (x[1][0],(x[0], x[1][1]))).join(model_rdd).map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1])))
    pred = user_sig_rdd.map(lambda x: find_topn(x, n_weights)).map(lambda x: calc_pred(x)).collect()
    
    with open(output_file, 'w') as outfile:
        for pair in pred:
            json_dict = {'user_id':pair[0], 'business_id':pair[1], 'stars':pair[2]}
            outfile.write(json.dumps(json_dict) + '\n')


def map_to_matrix(line):
    id = line[0]
    ratings_list = line[1]
    rating_output = []
    seen = []
    for rating in ratings_list:
        if rating[0] not in seen:
            rating_output.append(rating)
            seen.append(rating[0])
    return (id, rating_output)


def find_topn(line, n_weights):
    uid = line[0][0]
    bid = line[0][1]
    star_dict = {bid:stars for bid, stars in line[1][0]}
    corr_dict = {bid:corr for bid, corr in line[1][1]}
    return_list = []
    max_corr = [float('-inf') for _ in range(n_weights)]
    for bid in star_dict:
        if bid in corr_dict:
            for n in range(n_weights):
                if max_corr[n] < corr_dict[bid]:
                    max_corr[n] = corr_dict[bid]
                    if n >= len(return_list):
                        return_list.append((bid, star_dict[bid], corr_dict[bid]))
                    else:
                        return_list[n] = (bid, star_dict[bid], corr_dict[bid])
                    break

    return (uid, bid, return_list)


def calc_pred(line):
    uid = line[0]
    bid = line[1]
    data_list = line[2]
    numerator = 0.0
    denominator = 0.0
    for bid2, stars, corr in data_list:
        numerator += stars*corr
        denominator += abs(corr)
    if denominator != 0.0:
        pred = numerator / denominator
    else:
        pred = 0.0
    return (uid, bid, pred)

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
    parser.add_argument('--test_file', type=str, default='./data/test_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--output_file', type=str, default='./outputs/task2.case1.test.out')
    parser.add_argument('--time_file', type=str, default='./outputs/time.out')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))
