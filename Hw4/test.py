import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict

def main():
    graph = {'e':['d','f'],'d':['e','f','g','b'],'f':['e','d','g'],'g':['d','f','h'],'b':['d','a','c'],'a':['b','c'],'c':['b','a','i'],'h':['g','i'],'i':['h','c']}
    uid = 'e'
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
    scores = [*set(scores)]

    print(shortest_paths)
    print(scores)
    
    return

def betweenness_recursive(cur_node, graph, layers, shortest_paths, parent):
    children = [node for node in graph[cur_node] if layers[node] > layers[cur_node]]
    if len(children) == 0:
        val = shortest_paths[parent]/shortest_paths[cur_node]
        return val, [((*sorted((parent, cur_node)),) , val)]
    node_val = 1
    scores = []
    for child in children:
        val, score = betweenness_recursive(child, graph, layers, shortest_paths, cur_node)
        node_val += val
        scores.extend(score)
    if parent is not None:
        val = node_val * (shortest_paths[parent]/shortest_paths[cur_node])
        scores.append(((*sorted((parent, cur_node)),) , val))
    else:
        val = 0
    return val, scores

if __name__ == '__main__':
    main()