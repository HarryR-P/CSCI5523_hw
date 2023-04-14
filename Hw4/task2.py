import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict


def main(filter_threshold, input_file, output_file, betweenness_output_file, sc : pyspark.SparkContext):

    data_rdd = sc.textFile(input_file).filter(lambda x: x[0] != 'user_id,business_id').map(lambda x: (*(x.split(',')),))
    matrix_rdd = data_rdd.map(lambda x: (x[1],x[0])).aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).map(lambda x:(x[0],(*set(x[1]),)))
    graph = matrix_rdd.flatMap(map_co_thr).reduceByKey(lambda a,b: a+b).filter(lambda x: x[1] >= filter_threshold).flatMap(lambda x: [x[0],(x[0][1],x[0][0])])\
        .aggregateByKey([], lambda a,b: a + [b], lambda a,b: a + b).collectAsMap()
    betweenness_rdd = data_rdd.map(lambda x: x[0]).distinct().flatMap(lambda x: calc_betweenness(x, graph)).reduceByKey(lambda a,b: a+b).map(lambda x: (x[0],x[1]/2))
    betweenness = betweenness_rdd.sortBy(lambda x: (-x[1],x[0])).collect()



    with open(betweenness_output_file, 'w') as outfile:
        for line in betweenness:
            outfile.write(line + '\n')

    # example of identified communities
    # communities = [['23y0Nv9FFWn_3UWudpnFMA'],['3Vd_ATdvvuVVgn_YCpz8fw'], ['0KhRPd66BZGHCtsb9mGh_g', '5fQ9P6kbQM_E0dx8DL6JWA' ]]

    # for i in communities:
    #     print(i)

    # """ code for saving the output to file in the correct format """
    # resultDict = {}
    # for community in communities:
    #     community = list(map(lambda userId: "'" + userId + "'", sorted(community)))
    #     community = ", ".join(community)

    #     if len(community) not in resultDict:
    #         resultDict[len(community)] = []
    #     resultDict[len(community)].append(community)

    # results = list(resultDict.items())
    # results.sort(key = lambda pair: pair[0])

    # output = open(output_file, "w")

    # for result in results:
    #     resultList = sorted(result[1])
    #     for community in resultList:
    #         output.write(community + "\n")
    # output.close()


def map_co_thr(line):
    ratings_list = line[1]
    if len(ratings_list) < 2: return []
    pairs = combinations(ratings_list,2)
    return [(tuple(sorted(pair)),1) for pair in pairs]


def calc_betweenness(uid, graph):
    if uid not in graph: return []
    # BFS
    queue = [uid]
    layers = {uid: 0}
    shortest_paths = defaultdict(int)
    while queue:
        cur = queue.pop(0)
        cur_layer = layers[cur]
        shortest_paths[cur] += 1
        for child in graph[cur]:
            if child not in layers:
                layers[child] = cur_layer + 1
            if cur_layer < layers[child]:
                queue.append(child)
    shortest_paths = dict(shortest_paths)

    # DFS 
    _, scores = betweenness_recursive(uid, graph, layers, shortest_paths, None)
    return scores


def betweenness_recursive(cur_node, graph, layers, shortest_paths, parent):
    connected_nodes = graph[cur_node]
    children = [node for node in connected_nodes if layers[node] > layers[cur_node]]
    if len(children) == 0:
        return 1/shortest_paths[cur_node], [((*sorted((parent, cur_node)),) , 1/shortest_paths[cur_node])]
    node_val = 1
    scores = []
    for child in children:
        val, score = betweenness_recursive(child, graph, layers, shortest_paths, cur_node)
        node_val += val
        scores.extend(score)
    if parent is not None:
        scores.append(((*sorted((parent, cur_node)),) , node_val/shortest_paths[cur_node]))
    return node_val / shortest_paths[cur_node], scores


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
    parser.add_argument('--betweenness_output_file', type=str, default='./result.txt', help='the output file contains your answers')
    args = parser.parse_args()

    main(args.filter_threshold, args.input_file, args.community_output_file, args.betweenness_output_file, sc)
    sc.stop()



