import argparse
import json
import time
import pyspark
import math
from itertools import combinations
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def main(train_file, test_file, output_file, sc : pyspark.SparkContext):

    reviews_rdd = sc.textFile(train_file).map(lambda x: json.loads(x))
    users = reviews_rdd.map(lambda x: x['user_id']).distinct().collect()
    businesses = reviews_rdd.map(lambda x: x['business_id']).distinct().collect()
    business_label2idx = {business: idx for idx, business in enumerate(businesses)}
    business_idx2label = {idx: business for idx, business in enumerate(businesses)}

    users_label2idx = {user: idx for idx, user in enumerate(users)}
    users_idx2label = {idx: user for idx, user in enumerate(users)}

    ratings_rdd = reviews_rdd.map(lambda x: Rating(users_label2idx[x['user_id']], business_label2idx[x['business_id']], float(x['stars'])))

    model = ALS.train(ratings_rdd, rank=4, iterations=11, lambda_=0.07)
    #model.save(sc,model_file)

    test_rdd = sc.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (users_label2idx[x['user_id']],business_label2idx[x['business_id']]))
    print(test_rdd.count())
    predictions = model.predictAll(test_rdd).map(lambda x: (users_idx2label[x[0]], business_idx2label[x[1]], x[2])).collect()
    
    with open(output_file, 'w') as outfile:
        for pair in predictions:
            json_dict = {'user_id':pair[0], 'business_id':pair[1], 'stars':pair[2]}
            outfile.write(json.dumps(json_dict) + '\n')

    return

if __name__ == '__main__': 
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('cmop') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json')
    parser.add_argument('--test_file', type=str, default='./outputs/model')
    parser.add_argument('--output_file', type=str, default='./outputs/out.json')
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.output_file, sc)
    sc.stop()

    print('The run time is: ', (time.time() - start_time))