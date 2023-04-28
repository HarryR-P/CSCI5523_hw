import os
import argparse
import json
import time
import csv
import numpy as np
import math
import pandas as pd
import heapq
from itertools import combinations

def main():
    columns = ['id'] + [f'x{i}' for i in range(50)]
    data_df = pd.read_csv("C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\data\\test2\\data0.txt", names=columns).set_index('id')
    print(h_cluster(data_df.head(1000)))
    
    
    return


def h_cluster(data_df : pd.DataFrame, max_dist = 500, max_itter = 500):
    points_df = data_df.copy(deep=True)
    point2id = {(i,):i for i in points_df.index.tolist()}
    max_idx = len(point2id) - 1
    dist_queue = []
    for p1, p2 in combinations(point2id.keys(),2):
        dist = calc_dist(points_df.loc[point2id[p1]],points_df.loc[point2id[p2]])
        heapq.heappush(dist_queue, (dist,p1,p2))
    for _ in range(max_itter):
        dist, p1, p2 = heapq.heappop(dist_queue)
        if p1 == '-':
            continue
        if dist > max_dist:
            break

        for i, (cur_dist, cur_p1, cur_p2) in enumerate(dist_queue):
            if p1 == cur_p1 or p1 == cur_p2:
                dist_queue[i] = (cur_dist, '-', '-')
            elif p2 == cur_p1 or p2 == cur_p2:
                dist_queue[i] = (cur_dist, '-', '-')

        cluster_idx = (p1,p2)
        centroid = list(np.average([points_df.loc[point2id[p1]].to_numpy(),points_df.loc[point2id[p2]].to_numpy()],axis=0))
        columns = [f'x{i}' for i in range(points_df.shape[1]-1)]
        series_dict = {'id': max_idx + 1}
        for dim, name in zip(centroid,columns):
            series_dict[name] = dim
        row = pd.Series(series_dict).to_frame().T.set_index('id')
        points_df = pd.concat([points_df, row])
        point2id[cluster_idx] = max_idx + 1
        max_idx += 1
        del point2id[p1]
        del point2id[p2]
        for point in point2id.keys():
            if point == cluster_idx:
                continue
            dist = calc_dist(points_df.loc[point2id[point]],points_df.loc[point2id[cluster_idx]])
            heapq.heappush(dist_queue, (dist,point,cluster_idx))

    return [flatten(cluster) for cluster in point2id.keys()]


def calc_dist(p1, p2):
    dist = 0
    for d1, d2 in zip(p1,p2):
        dist += (d1 - d2)**2
    return np.sqrt(dist).item()


def flatten(nestedlist):
    if not(bool(nestedlist)):
        return nestedlist
    if isinstance(nestedlist[0], tuple):
        return flatten(*nestedlist[:1]) + flatten(nestedlist[1:])
    return nestedlist[:1] + flatten(nestedlist[1:])


def mahalanobis_dist(p1, p2, sd_list):
    dist = 0
    for d1, d2, sd in zip(p1, p2, sd_list):
        dist += ((d1 - d2) / sd)**2
    return np.sqrt(dist).item()


if __name__ == '__main__':
    main()