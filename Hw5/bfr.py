import os
import argparse
import json
import time
import numpy as np
import math
import pandas as pd
import heapq
from sklearn.cluster import KMeans
from itertools import combinations

def main(input_path, n_cluster, out_file1, out_file2):
    round_id = 1
    nof_compression_points = 0
    print(f'Round: {round_id}')
    dir_list = os.listdir(input_path)
    kmeans_path = dir_list.pop(0)
    data_df = pd.read_csv(os.path.join(input_path, kmeans_path),header=None)
    columns = ['id'] + [f'x{i}' for i in range(data_df.shape[1]-1)]
    columns_dict = {i: name for i, name in enumerate(columns)}
    data_df = data_df.rename(columns=columns_dict).set_index('id')
    intermediate_df = pd.DataFrame(columns=['round_id','nof_cluster_discard','nof_point_discard','nof_cluster_compression','nof_point_compression','nof_point_retained'])
    points_df = data_df.sample(frac=0.5)
    points_idx = points_df.index.tolist()
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto').fit(points_df.to_numpy())
    labels = kmeans.labels_
    return_dict = dict([])
    DS = []
    CS = []
    RS = []
    clusters = dict([])
    for idx, label in zip(points_df.index.tolist(),labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    for cluster in clusters.values():
        if len(cluster) > 1:
            row = {'N':0,'SUM':[0 for _ in range(data_df.shape[1])],'SUMSQ':[0 for _ in range(data_df.shape[1])]}
            for idx in cluster:
                row['N'] += 1
                row['SUM'] = np.sum([row['SUM'],data_df.loc[idx].to_numpy()],axis=0)
                row['SUMSQ'] = np.sum([row['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
            DS.append(row)
    
    print(len(DS))

    unlabled_list ,ds_points = calc_DS_points(data_df, DS)
    for idx, label in ds_points.items():
        return_dict[str(idx)] = int(label)
        DS[label]['N'] += 1
        DS[label]['SUM'] = np.sum([DS[label]['SUM'],data_df.loc[idx].to_numpy()],axis=0)
        DS[label]['SUMSQ'] = np.sum([DS[label]['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
    cluster_df = data_df.filter(items=unlabled_list, axis=0)

    CS_kmeans = KMeans(n_clusters=5*n_cluster, n_init='auto').fit(cluster_df.to_numpy())
    CS_labels = CS_kmeans.labels_
    clusters = dict([])
    for idx, label in zip(cluster_df.index.tolist(),CS_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    for group in clusters.values():
        if len(clusters) == 1:
            RS.append([group[0]] + cluster_df.loc[group[0]].values.tolist())
        else:
            cs_cluster = {'N':0,'SUM':[0 for _ in range(cluster_df.shape[1])],'SUMSQ':[0 for _ in range(cluster_df.shape[1])]}
            cs_points = []
            for idx in group:
                nof_compression_points += 1
                cs_points.append(str(idx))
                cs_cluster['N'] += 1
                cs_cluster['SUM'] = np.sum([cs_cluster['SUM'],cluster_df.loc[idx].to_numpy()],axis=0)
                cs_cluster['SUMSQ'] = np.sum([cs_cluster['SUMSQ'],np.square(cluster_df.loc[idx].to_numpy())],axis=0)
            CS.append((cs_cluster,cs_points))
    CS = merge_CS(CS)
    
    point_discard = len(labels)
    row = pd.Series({'round_id':round_id,'nof_cluster_discard':len(DS),'nof_point_discard':point_discard,'nof_cluster_compression':len(CS),
                         'nof_point_compression':nof_compression_points,'nof_point_retained':len(RS)})
    intermediate_df = pd.concat([intermediate_df, row.to_frame().T],ignore_index=True)
    round_id += 1
    
    for test_name in dir_list:
        print(f'Round: {round_id}')
        data_df = pd.read_csv(os.path.join(input_path, test_name), header=None)
        columns_dict = {i: name for i, name in enumerate(columns)}
        data_df = data_df.rename(columns=columns_dict).set_index('id')
        unlabled_list ,ds_points = calc_DS_points(data_df, DS)
        for idx, label in ds_points.items():
            return_dict[str(idx)] = int(label)
            DS[label]['N'] += 1
            DS[label]['SUM'] = np.sum([DS[label]['SUM'],data_df.loc[idx].to_numpy()],axis=0)
            DS[label]['SUMSQ'] = np.sum([DS[label]['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
        cluster_df = data_df.filter(items=unlabled_list, axis=0)
        
        for point in RS:
            if any(math.isnan(num) for num in point):
                print(point)
            series_dict = {name : dim for name,dim in zip(columns, point)}
            row = pd.Series(series_dict).to_frame().T.set_index('id')
            cluster_df = pd.concat([cluster_df, row])
        clusters = []
        print(cluster_df.shape[0])
        CS_kmeans = KMeans(n_clusters=5*n_cluster, n_init='auto').fit(cluster_df.to_numpy())
        clusters = dict([])
        for idx, label in zip(cluster_df.index.tolist(),CS_kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        RS = []
        for group in clusters.values():
            if len(group) == 1:
                RS.append([group[0]] + cluster_df.loc[group[0]].values.tolist())
            else:
                cs_cluster = {'N':0,'SUM':[0 for _ in range(cluster_df.shape[1])],'SUMSQ':[0 for _ in range(cluster_df.shape[1])]}
                cs_points = []
                for idx in group:
                    nof_compression_points += 1
                    cs_points.append(str(idx))
                    cs_cluster['N'] += 1
                    cs_cluster['SUM'] = np.sum([cs_cluster['SUM'],cluster_df.loc[idx].to_numpy()],axis=0)
                    cs_cluster['SUMSQ'] = np.sum([cs_cluster['SUMSQ'],np.square(cluster_df.loc[idx].to_numpy())],axis=0)
                CS.append((cs_cluster,tuple(cs_points)))
        CS = merge_CS(CS)
        point_discard += len(ds_points)
        row = pd.Series({'round_id':round_id,'nof_cluster_discard':len(DS),'nof_point_discard':point_discard,'nof_cluster_compression':len(CS),
                         'nof_point_compression':nof_compression_points,'nof_point_retained':len(RS)})
        intermediate_df = pd.concat([intermediate_df, row.to_frame().T],ignore_index=True)
        round_id += 1
    
    for point in RS:
        idx = point[0]
        return_dict[str(idx)] = -1

    intermediate_df.to_csv(out_file2, index=False)

    with open(out_file1,'w') as outfile:
        json.dump(return_dict,outfile)
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


def calc_DS_points(data_df : pd.DataFrame, DS, max_dist=4):
    point_idxs = data_df.index.tolist()
    dists = {idx: [] for idx in point_idxs}
    for cluster in DS:
        sd = np.sqrt((cluster['SUMSQ'] / cluster['N']) - np.square(cluster['SUM'] / cluster['N']))
        centroid = cluster['SUM'] / cluster['N']
        for idx in point_idxs:
            dist = mahalanobis_dist(data_df.loc[idx],centroid,sd)
            dists[idx].append(dist)
    del_list = []
    for k, dist_list in dists.items():
        min_val = min(dist_list)
        min_idx = np.argmin(dist_list)
        if min_val > max_dist*math.sqrt(data_df.shape[1]):
            del_list.append(k)
        else:
            dists[k] = min_idx
    for k in del_list:
        del dists[k]
    return del_list, dists


def merge_CS(CS, dims, max_itter=500, max_dist = 3):
    points_dict = {(i,):cluster for i, cluster in enumerate(CS)}
    priority_queue = []
    for c1, c2 in  combinations(points_dict.keys(),2):
        centroid1 = points_dict[c1][0]['SUM'] / points_dict[c1][0]['N']
        centroid2 = points_dict[c2][0]['SUM'] / points_dict[c2][0]['N']
        std = np.sqrt((points_dict[c1][0]['SUMSQ'] / points_dict[c1][0]['N']) - np.square(points_dict[c1][0]['SUM'] / points_dict[c1][0]['N']))
        dist = mahalanobis_dist(centroid1, centroid2, std)
        heapq.heappush(priority_queue,(dist,c1,c2))
    for _ in range(max_itter):
        if len(priority_queue) == 0:
            break
        dist, c1, c2 = heapq.heappop(priority_queue)
        if c1 == '-':
            continue
        if dist > max_dist*math.sqrt(dims):
            break
        for i, (cur_dist, cur_c1, cur_c2) in enumerate(priority_queue):
            if c1 == cur_c1 or c1 == cur_c2:
                priority_queue[i] = (cur_dist, '-', '-')
            elif c2 == cur_c1 or c2 == cur_c2:
                priority_queue[i] = (cur_dist, '-', '-')

        cluster_idx = (c1,c2)
        cluster_vals = {'N':points_dict[c1][0]['N'] + points_dict[c2][0]['N'],'SUM':points_dict[c1][0]['SUM'] + points_dict[c2][0]['SUM'],
                      'SUMSQ':points_dict[c1][0]['SUMSQ'] + points_dict[c2][0]['SUMSQ']}
        cluster_points = tuple(list(points_dict[c1][1]) + list(points_dict[c2][1]))
        points_dict[cluster_idx] = (cluster_vals,cluster_points)
        del points_dict[c1]
        del points_dict[c2]
        for point in points_dict.keys():
            if point == cluster_idx:
                continue
            centroid1 = points_dict[point][0]['SUM'] / points_dict[point][0]['N']
            centroid2 = points_dict[cluster_idx][0]['SUM'] / points_dict[cluster_idx][0]['N']
            std = np.sqrt((points_dict[point][0]['SUMSQ'] / points_dict[point][0]['N']) - np.square(points_dict[point][0]['SUM'] / points_dict[point][0]['N']))
            dist = mahalanobis_dist(centroid1, centroid2, std)
            heapq.heappush(priority_queue, (dist,point,cluster_idx))    
    return list(points_dict.values())


def mahalanobis_dist(p1, p2, sd_list):
    dist = 0
    for d1, d2, sd in zip(p1, p2, sd_list):
        if sd == 0:
            sd_list
        dist += ((d1 - d2) / sd)**2
    dist = np.sqrt(dist).item()
    if type(dist) != int:
        dist = 100
    return dist


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default='./test1', help='')
    parser.add_argument('--n_cluster', type=int, default=10, help='number clusters for k-means')
    parser.add_argument('--out_file1', type=str, default='./val.json', help='the output file contains your answers')
    parser.add_argument('--out_file2', type=str, default='./intermediate.csv', help='the output file contains your intermediate results')
    args = parser.parse_args()

    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2)
    print(f'Runtime: {time.time() - start_time}')