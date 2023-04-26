import os
import argparse
import json
import time
import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
from itertools import combinations

def main(input_path, n_cluster, out_file1, out_file2):
    dir_list = os.listdir(input_path)
    kmeans_path = dir_list.pop(0)
    columns = ['id'] + [f'x{i}' for i in range(10)]
    data_df = pd.read_csv(os.path.join(input_path, kmeans_path), names=columns).set_index('id')
    points = data_df.to_numpy()
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto').fit(points)
    labels = kmeans.labels_
    return_dict = dict([])
    DS = [{'N':0,'SUM':[0 for _ in data_df.iloc[0]],'SUMSQ':[0 for _ in data_df.iloc[0]]} for _ in range(n_cluster)]
    CS = []
    RS = []
    for idx, label in zip(data_df.index.tolist(),labels):
        return_dict[idx] = label
        DS[label]['N'] += 1
        DS[label]['SUM'] = np.sum([DS[label]['SUM'],data_df.iloc[idx].to_numpy()],axis=0)
        DS[label]['SUMSQ'] = np.sum([DS[label]['SUMSQ'],np.square(data_df.iloc[idx].to_numpy())],axis=0)

    for test_name in dir_list:
        columns = ['id'] + [f'x{i}' for i in range(10)]
        data_df = pd.read_csv(os.path.join(input_path, test_name), names=columns).set_index('id')


    
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


def calc_DS_points(data_df : pd.DataFrame, DS):
    point_idxs = data_df.index.tolist()
    dists = {idx: [] for idx in point_idxs}
    for cluster in DS:
        sd = np.sqrt(cluster['SUMSQ'] / cluster['N'] - np.square(cluster['SUM'] / cluster['N']))
        centroid = cluster['SUM'] / cluster['N']
        for idx in point_idxs:
            dist = mahalanobis_dist(data_df.iloc[idx],centroid,sd)
            dists[idx].append(dist)
    for k, d_list in dists.items():
        min_val = min(d_list)
        min_idx = np.argmin(d_list)
        if min_val > 2:
            del dists[k]
        else:
            dists[k] = min_idx
    return dists


def mahalanobis_dist(p1, p2, sd_list):
    dist = 0
    for d1, d2, sd in zip(p1, p2, sd_list):
        dist += ((d1 - d2) / sd)**2
    return np.sqrt(dist).item()


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default='./test1', help='')
    parser.add_argument('--n_cluster', type=int, default=5, help='number clusters for k-means')
    parser.add_argument('--out_file1', type=str, default='./val.json', help='the output file contains your answers')
    parser.add_argument('--out_file2', type=str, default='./intermediate.csv', help='the output file contains your intermediate results')
    args = parser.parse_args()

    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2)