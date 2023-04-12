import argparse
import json
import time
import pyspark
from itertools import combinations
from collections import defaultdict

def main():
    uid = 'e'
    graph = {'e':['d','f'], 'd':['e','f','b','g'], 'f':['e','d','g'], 'g':['d','f'], 'b':['d','a','c'], 'a':['b','c'], 'c':['b','a']}
    queue = [(None,[],uid)]
    visited = [uid]
    shortest_dist = dict([])
    shortest_num = defaultdict(int)
    while queue:
        parent, siblings, cur = queue.pop(0)
        if parent is None:
            shortest_dist[cur] = 0
            shortest_num[cur] += 1
        elif shortest_dist.setdefault(cur, float('inf')) >= shortest_dist[parent] + 1:
            shortest_dist[cur] = shortest_dist[parent] + 1
            shortest_num[cur] += 1
        for child in graph[cur]:
            if child != parent  and child not in siblings:
                queue.append((cur,set([node for node in graph[cur] if node not in siblings]),child))
            elif child != parent and shortest_dist.setdefault(child, float('inf')) == shortest_dist[cur] + 1:
                shortest_num[child] += 1
    
    print(dict(shortest_num))
    return

if __name__ == '__main__':
    main()