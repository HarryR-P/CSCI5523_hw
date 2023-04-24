import os
import argparse
import json
import time
from itertools import combinations

def main(input_path, n_cluster, out_file1, out_file2):
    dir_list = os.listdir(input_path)
    kmeans_path = dir_list.pop(0)
    with open(os.path.join(input_path,kmeans_path), 'w') as f:
        line = f.read()

    return


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default=7, help='')
    parser.add_argument('--n_cluster', type=int, default='./data/test1', help='number clusters for k-means')
    parser.add_argument('--out_file1', type=str, default='./val.json', help='the output file contains your answers')
    parser.add_argument('--out_file2', type=str, default='./intermediate.csv', help='the output file contains your intermediate results')
    args = parser.parse_args()

    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2)