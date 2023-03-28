import argparse
import json
import time
import pyspark
import findspark
from itertools import combinations

def main():
    sim = [('a','b',0.5), ('c','d', 0.8)]

    with open('C:\\Users\\harri\\Documents\\CSCI_5523_local\\CSCI5523_hw\\Hw3\\test.json', 'w') as outfile:
        for i, pair in enumerate(sim):
            json_dict = {'b1':pair[0], 'b2':pair[1], 'sim':pair[2]}
            json.dump(json_dict, outfile)
            outfile.write('\n')
    return

def jacobian(s1, s2):
    sim_cout = 0
    for el1, el2 in zip(s1, s2):
        if el1 == el2:
            sim_cout += 1
    return sim_cout / len(s1)

if __name__ == '__main__':
    main()