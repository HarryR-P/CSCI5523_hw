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
        data = list(csv.reader(f,delimiter=','))
        k_points = np.random.choice(len(data),size=n_cluster,replace=False)
        clusters = dict([])
        clusters_dist = dict([])
        for centroid in k_points:
            for point in data:
                point_idx = point[0]
                centroid_idx = centroid[0]
                distance = 0
                for dim_idx in range(1,len(point)):
                    distance += (point[dim_idx] - centroid[dim_idx])**2
                distance = math.sqrt(distance)
                if point_idx not in clusters:
                    clusters[point_idx] = centroid_idx
                    clusters_dist[point_idx] = distance
                elif distance < clusters_dist[point_idx]:
                    clusters[point_idx] = centroid_idx
                    clusters_dist[point_idx] = distance
    print(clusters)

    return


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default='./test1', help='')
    parser.add_argument('--n_cluster', type=int, default=5, help='number clusters for k-means')
    parser.add_argument('--out_file1', type=str, default='./val.json', help='the output file contains your answers')
    parser.add_argument('--out_file2', type=str, default='./intermediate.csv', help='the output file contains your intermediate results')
    args = parser.parse_args()

    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2)