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
    #data_df = pd.read_csv("C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\data\\test2\\data0.txt", header=None).set_index(0)
    l = [1,1]
    print(type(1)==int)
    
    
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


if __name__ == '__main__':
    main()