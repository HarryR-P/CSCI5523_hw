import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict

def main():
    uid = 'e'
    t = (1,2)
    print(t + '\n')
    
    return

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
    main()