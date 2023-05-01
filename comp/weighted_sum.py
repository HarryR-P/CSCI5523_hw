import argparse
import json
import time
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def main(rec1_file, rec2_file, output_file):
    pred1_dict = dict([])
    with open(rec1_file) as f:
        for line in f:
            input_json = json.loads(line)
            pred1_dict[(input_json['user_id'], input_json['business_id'])] = float(input_json['stars'])

    pred2_dict = dict([])
    with open(rec2_file) as f:
        for line in f:
            input_json = json.loads(line)
            pred2_dict[(input_json['user_id'], input_json['business_id'])] = float(input_json['stars'])

    keys = keys = list(pred1_dict.keys())

    return_dict = dict([])
    for key in keys:
        return_dict[key] = 0.4 * pred1_dict[key] + 0.6 * pred2_dict[key]
    
    with open(output_file, 'w') as outfile:
        for (id1, id2), val in return_dict.items():
            json_dict = {'user_id':id1, 'business_id':id2, 'stars':val}
            outfile.write(json.dumps(json_dict) + '\n')

if __name__ == '__main__': 
    start_time = time.time()

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--rec1_file', type=str, default='./data/recomender1_out.json')
    parser.add_argument('--rec2_file', type=str, default='./outputs/recomender2_out')
    parser.add_argument('--output_file', type=str, default='./outputs/out.json')
    args = parser.parse_args()

    main(args.rec1_file, args.rec2_file, args.output_file)

    print('The run time is: ', (time.time() - start_time))