import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict

def main():
    uid = 'e'
    graph = {'e':['d','f'], 'd':['e','f','b','g'], 'f':['e','d','g'], 'g':['d','f'], 'b':['d','a','c'], 'a':['b','c'], 'c':['b','a']}
    queue = [uid]
    layers = {uid: 0}
    shortest_num = defaultdict(int)
    while queue:
        cur = queue.pop(0)
        cur_layer = layers[cur]
        shortest_num[cur] += 1
        for child in graph[cur]:
            if child not in layers:
                layers[child] = cur_layer + 1
            if cur_layer < layers[child]:
                queue.append(child)
    shortest_num = dict(shortest_num)
    
    _, scores = betweenness_recursive(uid, graph, layers, shortest_num, None)
    
    print(scores)
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