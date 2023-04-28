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
    print(f'Round: {round_id}')
    dir_list = os.listdir(input_path)
    kmeans_path = dir_list.pop(0)
    data_df = pd.read_csv(os.path.join(input_path, kmeans_path),header=None)
    columns = ['id'] + [f'x{i}' for i in range(data_df.shape[1]-1)]
    columns = {i: name for i, name in enumerate(columns)}
    data_df = data_df.rename(columns=columns).set_index('id')
    intermediate_df = pd.DataFrame(columns=['round_id','nof_cluster_discard','nof_point_discard','nof_cluster_compression','nof_point_compression','nof_point_retained'])
    points = data_df.to_numpy()
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto').fit(points)
    labels = kmeans.labels_
    return_dict = dict([])
    DS = []
    CS = []
    RS = []
    clusters = [[] for _ in range(n_cluster)]
    for idx, label in zip(data_df.index.tolist(),labels):
        clusters[label].append(idx)
    print(clusters)
    label_idx = 0
    for cluster in clusters:
        if len(cluster) == 1:
            RS.append([cluster[0]] + data_df.loc[cluster[0]].values.tolist())
        elif len(cluster) > 1:
            row = {'N':0,'SUM':[0 for _ in range(data_df.shape[1])],'SUMSQ':[0 for _ in range(data_df.shape[1])]}
            for idx in cluster:
                return_dict[int(idx)] = int(label_idx)
                row['N'] += 1
                row['SUM'] = np.sum([row['SUM'],data_df.loc[idx].to_numpy()],axis=0)
                row['SUMSQ'] = np.sum([row['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
            label_idx += 1
            DS.append(row)
    print(return_dict)

    # for idx, label in zip(data_df.index.tolist(),labels):
    #     return_dict[int(idx)] = int(label)
    #     DS[label]['N'] += 1
    #     DS[label]['SUM'] = np.sum([DS[label]['SUM'],data_df.loc[idx].to_numpy()],axis=0)
    #     DS[label]['SUMSQ'] = np.sum([DS[label]['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
    point_discard = len(labels)
    row = pd.Series({'round_id':round_id,'nof_cluster_discard':len(DS),'nof_point_discard':point_discard,'nof_cluster_compression':len(CS),
                         'nof_point_compression':0,'nof_point_retained':len(RS)})
    intermediate_df = pd.concat([intermediate_df, row.to_frame().T],ignore_index=True)
    round_id += 1
    # for test_name in dir_list:
    
    for test_name in dir_list:
        print(f'Round: {round_id}')
        data_df = pd.read_csv(os.path.join(input_path, test_name), header=None)
        columns = ['id'] + [f'x{i}' for i in range(data_df.shape[1]-1)]
        columns = {i: name for i, name in enumerate(columns)}
        data_df = data_df.rename(columns=columns).set_index('id')
        unlabled_list ,ds_points = calc_DS_points(data_df, DS)
        for idx, label in ds_points.items():
            return_dict[int(idx)] = int(label)
            DS[label]['N'] += 1
            DS[label]['SUM'] = np.sum([DS[label]['SUM'],data_df.loc[idx].to_numpy()],axis=0)
            DS[label]['SUMSQ'] = np.sum([DS[label]['SUMSQ'],np.square(data_df.loc[idx].to_numpy())],axis=0)
        data_df = data_df.filter(items=unlabled_list, axis=0)
        cluster_df = data_df.copy(deep=True)
        for point in RS:
            row = pd.Series({'id':point[0],'x0':point[1],'x1':point[2],'x2':point[3],'x3':point[4],'x4':point[5],'x5':point[6],'x6':point[7],'x7':point[8],'x8':point[9],
                            'x9':point[10]}).to_frame().T.set_index('id')
            cluster_df = pd.concat([cluster_df, row])
        i = 0
        clusters = []
        print(cluster_df.shape[0])
        while i < cluster_df.shape[0]:
            clusters += h_cluster(cluster_df.iloc[i:i+1000])
            i += 1000
        RS = []
        nof_compression_points = 0
        for group in clusters:
            if len(group) == 1:
                RS.append([group[0]] + cluster_df.loc[group[0]].values.tolist())
            else:
                cs_cluster = {'N':0,'SUM':[0 for _ in range(cluster_df.shape[1])],'SUMSQ':[0 for _ in range(cluster_df.shape[1])]}
                for idx in group:
                    nof_compression_points += 1
                    return_dict[int(idx)] = -1
                    cs_cluster['N'] += 1
                    cs_cluster['SUM'] = np.sum([cs_cluster['SUM'],cluster_df.loc[idx].to_numpy()],axis=0)
                    cs_cluster['SUMSQ'] = np.sum([cs_cluster['SUMSQ'],np.square(cluster_df.loc[idx].to_numpy())],axis=0)
                CS.append(cs_cluster)
        CS = merge_CS(CS)
        point_discard += len(ds_points)
        row = pd.Series({'round_id':round_id,'nof_cluster_discard':len(DS),'nof_point_discard':point_discard,'nof_cluster_compression':len(CS),
                         'nof_point_compression':nof_compression_points,'nof_point_retained':len(RS)})
        intermediate_df = pd.concat([intermediate_df, row.to_frame().T],ignore_index=True)
        round_id += 1
    
    for point in RS:
        idx = point[0]
        return_dict[int(idx)] = -1

    intermediate_df.to_csv(out_file2, index=False)

    with open(out_file1,'w') as outfile:
        json.dump(return_dict,outfile)
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
        if len(dist_queue) == 0:
            break
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


def calc_DS_points(data_df : pd.DataFrame, DS, max_dist=5):
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
        if min_val > max_dist:
            del_list.append(k)
        else:
            dists[k] = min_idx
    for k in del_list:
        del dists[k]
    return del_list, dists


def merge_CS(CS, max_itter=500, max_dist = 3):
    points_dict = {(i,):cluster for i, cluster in enumerate(CS)}
    priority_queue = []
    for c1, c2 in  combinations(points_dict.keys(),2):
        centroid1 = points_dict[c1]['SUM'] / points_dict[c1]['N']
        centroid2 = points_dict[c2]['SUM'] / points_dict[c2]['N']
        std1 = np.sqrt((points_dict[c1]['SUMSQ'] / points_dict[c1]['N']) - np.square(points_dict[c1]['SUM'] / points_dict[c1]['N']))
        std2 = np.sqrt((points_dict[c2]['SUMSQ'] / points_dict[c2]['N']) - np.square(points_dict[c2]['SUM'] / points_dict[c2]['N']))
        dist = mahalanobis_dist(centroid1, centroid2, np.multiply(std1,std2))
        heapq.heappush(priority_queue,(dist,c1,c2))
    for _ in range(max_itter):
        if len(priority_queue) == 0:
            break
        dist, c1, c2 = heapq.heappop(priority_queue)
        if c1 == '-':
            continue
        if dist > max_dist:
            break
        for i, (cur_dist, cur_c1, cur_c2) in enumerate(priority_queue):
            if c1 == cur_c1 or c1 == cur_c2:
                priority_queue[i] = (cur_dist, '-', '-')
            elif c2 == cur_c1 or c2 == cur_c2:
                priority_queue[i] = (cur_dist, '-', '-')

        cluster_idx = (c1,c2)
        cluster_vals = {'N':points_dict[c1]['N'] + points_dict[c2]['N'],'SUM':points_dict[c1]['SUM'] + points_dict[c2]['SUM'],
                      'SUMSQ':points_dict[c1]['SUMSQ'] + points_dict[c2]['SUMSQ']}
        points_dict[cluster_idx] = cluster_vals
        del points_dict[c1]
        del points_dict[c2]
        for point in points_dict.keys():
            if point == cluster_idx:
                continue
            centroid1 = points_dict[point]['SUM'] / points_dict[point]['N']
            centroid2 = points_dict[cluster_idx]['SUM'] / points_dict[cluster_idx]['N']
            std1 = np.sqrt((points_dict[point]['SUMSQ'] / points_dict[point]['N']) - np.square(points_dict[point]['SUM'] / points_dict[point]['N']))
            std2 = np.sqrt((points_dict[cluster_idx]['SUMSQ'] / points_dict[cluster_idx]['N']) - np.square(points_dict[cluster_idx]['SUM'] / points_dict[cluster_idx]['N']))
            dist = mahalanobis_dist(centroid1, centroid2, std1 * std2)
            heapq.heappush(priority_queue, (dist,point,cluster_idx))    
    return list(points_dict.values())


def mahalanobis_dist(p1, p2, sd_list):
    dist = 0
    for d1, d2, sd in zip(p1, p2, sd_list):
        dist += ((d1 - d2) / sd)**2
    if math.isnan(dist):
        print(sd_list)
    return np.sqrt(dist).item()


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