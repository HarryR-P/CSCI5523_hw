import os
import argparse
import json
import time
import csv
import numpy as np
import math
import pandas as pd
import heapq
from sklearn.metrics import normalized_mutual_info_score
from itertools import combinations

def main():
    with open('C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\Hw5\\clusters.json') as f:
        pred_dict = json.load(f)
        pred_labels = []
        for i in range(len(pred_dict)):
            pred_labels.append(pred_dict[str(i)])
    with open('C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\data\\cluster1.json') as f:
        true_dict = json.load(f)
        true_labels = []
        for i in range(len(true_dict)):
            true_labels.append(true_dict[str(i)])
    
    print(normalized_mutual_info_score(true_labels, pred_labels))
    
    return

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


def merge_DS_CS(DS, CS, dims, max_dist = 4):
    dists = dict([])
    for ds_cluster in DS:
        ds_centroid = ds_cluster['SUM'] / ds_cluster['N']
        ds_std = np.sqrt((ds_cluster['SUMSQ'] / ds_cluster['N']) - np.square(ds_cluster['SUM'] / ds_cluster['N']))
        for cs_cluster, points in CS:
            cs_centroid = cs_cluster['SUM'] / cs_cluster['N']
            dist = mahalanobis_dist(cs_centroid, ds_centroid, ds_std)
            for point in points:
                if point not in dists:
                    dists[point] = []
                dists[point].append(dist)
    for point, dist_list in dists.items():
        min_val = min(dist_list)
        min_idx = np.argmin(dist_list)
        if min_val > max_dist*math.sqrt(dims):
            dists[point] = -1
        else:
            dists[point] = min_idx
    return dists


if __name__ == '__main__':
    main()