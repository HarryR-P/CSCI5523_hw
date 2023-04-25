import os
import argparse
import json
import time
import csv
import numpy as np
import math
from itertools import combinations

def main():
    a = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
    b = np.zeros(a.shape)
    print(a > b)
    return

if __name__ == '__main__':
    main()