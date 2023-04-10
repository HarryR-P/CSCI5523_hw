import argparse
import json
import time
import pyspark


def main(filter_threshold, input_file, output_file, sc):



    """ You need to write your own code """


    # example of identified communities
    communities = [['23y0Nv9FFWn_3UWudpnFMA'],['3Vd_ATdvvuVVgn_YCpz8fw'], ['0KhRPd66BZGHCtsb9mGh_g', '5fQ9P6kbQM_E0dx8DL6JWA' ]]

    for i in communities:
        print(i)

    """ code for saving the output to file in the correct format """
    resultDict = {}
    for community in communities:
        community = list(map(lambda userId: "'" + userId + "'", sorted(community)))
        community = ", ".join(community)

        if len(community) not in resultDict:
            resultDict[len(community)] = []
        resultDict[len(community)].append(community)

    results = list(resultDict.items())
    results.sort(key = lambda pair: pair[0])

    output = open(output_file, "w")

    for result in results:
        resultList = sorted(result[1])
        for community in resultList:
            output.write(community + "\n")
    output.close()



if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--filter_threshold', type=int, default=7, help='')
    parser.add_argument('--input_file', type=str, default='./ub_sample_data.csv', help='the input file')
    parser.add_argument('--community_output_file', type=str, default='./result.txt', help='the output file contains your answers')
    args = parser.parse_args()

    main(args.filter_threshold, args.input_file, args.community_output_file, sc)
    sc.stop()



