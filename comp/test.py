import argparse
import json
import time
import pyspark
import math
from sklearn.metrics import mean_squared_error
from itertools import combinations
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def main():
    pred_dict = dict([])
    with open('C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\comp\\test\\out.json') as f:
        for line in f:
            input_json = json.loads(line)
            pred_dict[(input_json['user_id'], input_json['business_id'])] = float(input_json['stars'])
    
    true_dict = dict([])
    with open('C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\data\\val_review_ratings.json') as f:
        for line in f:
            input_json = json.loads(line)
            true_dict[(input_json['user_id'], input_json['business_id'])] = float(input_json['stars'])

    keys = list(true_dict.keys())
    pred = []
    true = []
    pred125 = []
    true123 = []
    for key in keys:
        pred.append(pred_dict[key])
        true.append(true_dict[key])
        if true_dict[key] == 1.0 or true_dict[key] == 2.0 or true_dict[key] == 5.0:
            pred125.append(pred_dict[key])
            true123.append(true_dict[key])

    rmse = mean_squared_error(true, pred, squared=False)
    rmse125 = mean_squared_error(true123, pred125, squared=False)
    print(f'rmse: {rmse}')
    print(f'rmse125: {rmse125}')
    return

if __name__ == '__main__':
    main()