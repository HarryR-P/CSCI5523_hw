import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict

def main():
    graph = {'g':['h','e','f'],'h':['g','e'],'e':['g','h','f','d'],'f':['g','e'],'d':['e','a','c'],'a':['d','c','b'],'c':['d','a','b'],'b':['a','c']}
    uid = 'h'
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

    _, scores = betweenness_recursive(uid, graph, layers, shortest_paths, None)

    print(shortest_paths)
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
        score = node_val * (shortest_paths[parent]/shortest_paths[cur_node])
        scores.append(((*sorted((parent, cur_node)),) , score))
    else:
        score = 0
    return score, scores

if __name__ == '__main__':
    main()