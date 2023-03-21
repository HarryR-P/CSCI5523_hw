import pyspark
import argparse
import json
import time
# import findspark
from pyspark.sql import SparkSession


def main(args):
    start_time = time.time()
    # findspark.init()

    sc_conf = pyspark.SparkConf() \
        .setAppName('task3') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executer.memory', '4g')
    
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    # spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()

    review_path = args.input_file
    output_path = args.output_file
    n = args.n

    # review_rdd = spark.read.json(review_path).rdd
    review_rdd = sc.textFile(review_path).map(lambda x: json.loads(x))

    map_rdd = review_rdd.map(lambda x: (x['business_id'], 1))

    count_per_part = map_rdd.mapPartitions(count_in_partition).collect()

    count_rdd = map_rdd.reduceByKey(lambda a,b: a+b)
    filter_rdd = count_rdd.filter(lambda x: x[1] > n)

    return_dict = {"n_partitions": len(count_per_part), "n_items": count_per_part, "result" : filter_rdd.collect()}

    with open(output_path, 'w') as outfile:
        json.dump(return_dict, outfile)

    # spark.catalog.clearCache()
    sc.stop()
    print(f'Runtime: {time.time() - start_time}')


def count_in_partition(iterator):
  yield sum(1 for _ in iterator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A1T3')
    parser.add_argument('--input_file', type=str, default='./data/hw1/review.json', help='the input file')
    parser.add_argument('--output_file', type=str, default='./data/hw1/a1t3_default.json',
                        help='the output file that contains your answers')
    parser.add_argument('--n', type=int, default=10, help='buisnesses with more than n reviews')

    args = parser.parse_args()
    main(args)