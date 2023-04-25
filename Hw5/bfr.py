import os
import argparse
import json
import time
import csv
import numpy as np
import math
from itertools import combinations

def main(input_path, n_cluster, out_file1, out_file2):
    dir_list = os.listdir(input_path)
    kmeans_path = dir_list.pop(0)
    with open(os.path.join(input_path,kmeans_path)) as f:
        data = {int(line[0]) : [float(point) for point in line[1:]] for line in csv.reader(f,delimiter=',')}
    centroids = kmeans(data, n_cluster)
    print(centroids)


    return


def kmeans(data, n_cluster, max_itter = 100):
    k_points = np.random.choice(list(data.keys()),size=n_cluster,replace=False)
    centroids = np.array([data[idx] for idx in k_points])
    prev_centroids = np.zeros(centroids.shape)
    for _ in range(max_itter):
        clusters = {i : [] for i in data.keys()}
        for centroid in centroids:
            for point_idx, point in data.items():
                distance = 0
                for point_dim, centroid_dim in zip(point,centroid):
                    distance += (point_dim - centroid_dim)**2
                distance = math.sqrt(distance)
                clusters[point_idx].append(distance)
        for k, dist_list in clusters.items():
            cluster = np.argmin(dist_list)
            clusters[k] = cluster.item()

        cluster_dict = seperate_clusters(clusters, data, n_cluster)
        prev_centroids = centroids
        centroids = np.array([np.average(cluster,axis=0) for cluster in cluster_dict.values()])
        if np.any(np.abs(centroids - prev_centroids) > 0.01*np.ones(centroids.shape)):
            break
    return centroids


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