import pyspark
import argparse
import re
import json
#import findspark
from pyspark.sql import SparkSession


def main(args):

    #findspark.init()

    sc_conf = pyspark.SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executer.memory', '4g')
    
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    # spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()


    input_path = args.input_file
    output_path = args.output_file
    stopword_path = args.stopwords
    year = args.y
    m_users = args.m
    n_words = args.n

    return_dict = {}

    with open(stopword_path) as f:
        data = f.read()
    stopwords = data.replace('\n',' ').split()

    # review_rdd = spark.read.json(input_path).rdd
    review_rdd = sc.textFile(input_path).map(lambda x: json.loads(x))

    return_dict['A'] = review_rdd.count()
    review_rdd.unpersist()

    year_filter = review_rdd.filter(lambda x: int(x['date'][0:4])  == year)
    return_dict['B'] = year_filter.count()
    year_filter.unpersist()

    distinct_map = review_rdd.map(lambda x: (x['user_id'], 1))
    distinct_reduce = distinct_map.reduceByKey(lambda a,b: a+b)
    return_dict['C'] = distinct_reduce.count()

    sum_map = distinct_reduce.sortBy(lambda x: (-x[1], x[0]))
    return_dict['D'] = sum_map.take(m_users)

    distinct_reduce.unpersist()
    sum_map.unpersist()

    cout_map = review_rdd.flatMap(lambda x: filter_words(x, stopwords))
    count_reduce = cout_map.reduceByKey(lambda a,b: a+b)
    sorted_count = count_reduce.map(lambda x: ((-x[1], x[0]), x[0])).sortByKey()
    # sorted_count = count_reduce.sortByKey().sortBy(lambda x: x[1], ascending=False)
    top_words = sorted_count.take(n_words)
    return_dict['E'] = [x[1] for x in top_words]
    
    with open(output_path, 'w') as outfile:
        json.dump(return_dict, outfile)

    #spark.catalog.clearCache()
    sc.stop()

    return

def filter_words(row, stopwords):
    text = row['text']
    filterd_text = re.sub(r'[\(\[,\.!\?:;\}\)]','', text)

    filterd_words = filterd_text.split()
    return_words = [(word.lower(), 1) for word in filterd_words if word.lower() not in stopwords]

    return return_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_file', type=str, default='./data/hw1/review.json', help='the input file')
    parser.add_argument('--output_file', type=str, default='./data/hw1/a1t1.json',
                        help='the output file that contains your answers')
    parser.add_argument('--stopwords', type=str, default='.data/hw1/stopwords',
                        help='the file that contains stopwords')
    parser.add_argument('--y', type=int, default=2018, help='year')
    parser.add_argument('--m', type=int, default=10, help='top m users')
    parser.add_argument('--n', type=int, default=10, help='top n frequent words')

    args = parser.parse_args()
    main(args)