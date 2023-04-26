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
    points = data_df.drop(columns=['id']).to_numpy()
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto').fit(points)
    centroids = kmeans.cluster_centers_
    
    return


def h_cluster(data_df : pd.DataFrame, max_dist = 20):
    points = data_df.to_dict(orient='list')
    while len(points) > 1:
        min_dist = float('inf')
        min_points = -1
        for p1, p2 in combinations(points.keys(),2):
            dist = calc_dist(points[p1],points[p2])
            if min_dist > dist:
                min_dist = dist
                min_points = (*sorted(p1,p2),)
        if min_dist > max_dist:
            break
        centroid = list(np.average([points[min_points[0]],points[min_points[1]]],axis=0))
        points[min_points] = centroid
        del points[min_points[0]]
        del points[min_points[1]]

    return points.keys()


def calc_dist(p1, p2):
    dist = 0
    for d1, d2 in zip(p1,p2):
        dist += (d1 - d2)**2
    return np.sqrt(dist).item()


def seperate_clusters(clusters, data, n_cluster):
    return_dict = {i : [] for i in range(n_cluster)}
    for point_idx, centroid_idx in clusters.items():
        return_dict[centroid_idx].append(data[point_idx])
    return return_dict


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default='./test1', help='')
    parser.add_argument('--n_cluster', type=int, default=5, help='number clusters for k-means')
    parser.add_argument('--out_file1', type=str, default='./val.json', help='the output file contains your answers')
    parser.add_argument('--out_file2', type=str, default='./intermediate.csv', help='the output file contains your intermediate results')
    args = parser.parse_args()

    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2)