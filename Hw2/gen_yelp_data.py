import pyspark
import argparse
import json
import gc
import itertools
import time
import findspark
import pandas as pd
from pyspark.sql import SparkSession

def main():

    buisness_pd = pd.read_json('C:\\Users\\harri\\Documents\\CSCI_5523_local\\data\\business.json', lines=True)
    review_pd = pd.read_json('C:\\Users\\harri\\Documents\\CSCI_5523_local\\data\\review.json', lines=True)
    buisness_pd['business_id'] = buisness_pd['business_id'].astype(str)
    review_pd['business_id'] = review_pd['business_id'].astype(str)

    buisness_pd = buisness_pd[buisness_pd.state == 'NV']
    buisness_pd = buisness_pd.filter(items=['business_id'])
    review_pd = review_pd.filter(items=['user_id', 'business_id'])
    join_df = review_pd.merge(buisness_pd, how='inner', on='business_id')
    join_df.to_csv('C:\\Users\\harri\\Documents\\CSCI_5523_local\\data\\yelp.csv', sep=',', index=False)
    

    

if __name__ == '__main__':
    main()