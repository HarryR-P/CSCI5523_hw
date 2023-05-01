import argparse
import json
import time
import pyspark



def main(train_file, test_file, model_file, output_file, n_weights, sc : pyspark.SparkContext):
    review_rdd = sc.textFile(train_file).map(lambda x: json.loads(x))
    test_rdd = sc.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'],x['user_id']))
    model_rdd = sc.textFile(model_file).map(lambda x: json.loads(x)).flatMap(lambda x: [(x['b1'],(x['b2'],x['sim'])),(x['b2'],(x['b1'],x['sim']))]) \
        .distinct().aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b)
    stars_dict = review_rdd.map(lambda x: (x['user_id'],(x['business_id'], x['stars']))).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x: map_to_matrix(x)).collectAsMap()
    pred = test_rdd.leftOuterJoin(model_rdd).map(lambda x: find_topn(x, stars_dict, n_weights)).collect()
    
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


def find_topn(line, stars_dict, n_weights):
    uid = line[1][0]
    b1 = line[0]
    stars = stars_dict[uid]
    if line[1][1] is None:
        sims = []
    else:
        sims = line[1][1]
    star_dict = {bid:stars for bid, stars in stars}
    corr_dict = {bid:corr for bid, corr in sims}
    n_list = []
    for bid in star_dict:
        if bid in corr_dict:
            n_list.append((bid, star_dict[bid], corr_dict[bid]))

    n_list.sort(key=lambda x: x[2], reverse=True)
    return_list = n_list[:n_weights]
    
    if len(return_list) > 0:
        numerator = 0.0
        denominator = 0.0
        for bid2, stars, corr in return_list:
            numerator += stars*corr
            denominator += abs(corr)
        if denominator != 0.0:
            pred = numerator / denominator
        else:
            pred = 1.0     
    else:
        pred = sum(star_dict.values()) / len(star_dict)
    return (uid, b1, max(pred,1.0))


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
    parser.add_argument('--test_file', type=str, default='./data/test_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--output_file', type=str, default='./outputs/task2.case1.test.out')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    print('The run time is: ', (time.time() - start_time))
