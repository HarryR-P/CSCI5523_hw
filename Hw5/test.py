import os
import argparse
import json
import time
import csv
import numpy as np
import math
import pandas as pd
from itertools import combinations

def main():
    # columns = ['id'] + [f'x{i}' for i in range(10)]
    # data_df = pd.read_csv("C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\data\\test1\\data0.txt", names=columns).set_index('id')
    l = np.array([-2,2,2,2,2])
    print(np.square(l))
    
    return


def h_cluster(data_df : pd.DataFrame, max_dist = 20):
    points_df = data_df.copy(deep=True)
    point2id = {(i,):i for i in points_df.index.tolist()}
    max_idx = len(point2id) - 1
    while len(point2id) > 1:
        min_dist = float('inf')
        min_points = -1
        for p1, p2 in combinations(point2id.keys(),2):
            dist = calc_dist(points_df.iloc[point2id[p1]],points_df.iloc[point2id[p2]])
            if min_dist > dist:
                min_dist = dist
                min_points = (p1,p2)
        if min_dist > max_dist:
            break
        centroid = list(np.average([points_df.iloc[point2id[min_points[0]]].to_numpy(),points_df.iloc[point2id[min_points[1]]].to_numpy()],axis=0))
        
        row = pd.Series({'id':max_idx + 1,'x0':centroid[0],'x1':centroid[1],'x2':centroid[2],'x3':centroid[3],'x4':centroid[4],'x5':centroid[5],'x6':centroid[6],'x7':centroid[7],'x8':centroid[8],
                         'x9':centroid[9]}).to_frame().T.set_index('id')
        points_df = pd.concat([points_df, row])
        point2id[min_points] = max_idx + 1
        for point in min_points:
            del point2id[point]
        max_idx += 1

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


if __name__ == '__main__':
    main()