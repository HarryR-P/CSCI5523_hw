import pyspark
import argparse
import json
#import findspark
from pyspark.sql import SparkSession


def main(args):
    #findspark.init()

    sc_conf = pyspark.SparkConf() \
        .setAppName('task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executer.memory', '4g')
    
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    #spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()

    review_path = args.review_file
    business_path = args.business_file
    output_path = args.output_file
    top_n = args.n

    # review_rdd = spark.read.json(review_path).rdd
    # business_rdd = spark.read.json(business_path).rdd
    review_rdd = sc.textFile(review_path).map(lambda x: json.loads(x))
    business_rdd = sc.textFile(business_path).map(lambda x: json.loads(x))
    business_rdd = business_rdd.filter(lambda x: x['categories'] is not None)

    reviewByKey = review_rdd.map(lambda x: (x['business_id'], x['stars']))
    businessByKey = business_rdd.map(lambda x: (x['business_id'], x['categories']))

    join_rdd = reviewByKey.join(businessByKey)
    #print(join_rdd.take(10))
    cat_map = join_rdd.flatMap(lambda x: mapfunc(x))
    
    running_rdd = cat_map.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),
                                         lambda a,b: (a[0] + b[0], a[1] + b[1]))
    
    mean_rdd = running_rdd.mapValues(lambda v: v[0]/v[1])
    # sorted_rdd = mean_rdd.sortBy(lambda x: x[1], ascending=False)
    sorted_rdd = mean_rdd.map(lambda x: ((-x[1], x[0]), x)).sortByKey()

    return_list = sorted_rdd.take(top_n)

    return_dict = {"result" : [x[1] for x in return_list]}

    with open(output_path, 'w') as outfile:
        json.dump(return_dict, outfile)

    # spark.catalog.clearCache()
    sc.stop()

    return

def mapfunc(x):
    stars = x[1][0]
    catagories = x[1][1]
    cat_list = catagories.split(',')
    cat_list = [cat.strip() for cat in cat_list]

    return [ (catagory, stars) for catagory in cat_list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A1T2')
    parser.add_argument('--review_file', type=str, default='./data/hw1/review.json', help='the review file')
    parser.add_argument('--business_file', type=str, default='./data/hw1/business.json', help='the business file')
    parser.add_argument('--output_file', type=str, default='./data/hw1/a1t2.json',
                        help='the output file that contains your answers')
    parser.add_argument('--n', type=int, default=10, help='top n catagories with the most stars')

    args = parser.parse_args()
    main(args)