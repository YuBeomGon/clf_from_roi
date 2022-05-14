# this is for making csv file for classification, 6 class

import os
import argparse
import pandas as pd
import numpy as np
import sys
import cv2
import re

from utils import CLASS_MAPPER, convert, drop_wrong, paps_data_split

parser = argparse.ArgumentParser(description='get dataframe of annotations')
parser.add_argument('--saved_path', default='saved/', type=str, metavar='M',
                    help='path for saved results')
parser.add_argument('--ratio', default=0.25, type=float, metavar='M',
                    help='ratio')
parser.add_argument('--seed', default=1, type=int, metavar='M',
                    help='seed for re occurence')
parser.add_argument('--whole_slide', default=True, type=bool, metavar='M',
                    help='split by wsi or bbox')

if __name__ == "__main__":
    args = parser.parse_args()
    
    df = pd.read_csv(args.saved_path + 'df.csv')
    org_size = df.shape

    df['label'] = df.label.apply(lambda x : CLASS_MAPPER[str(x)])
    
    df = drop_wrong(df)
    normal_size = df.shape
    
    print('org_size {} normal size {} '.format(org_size, normal_size))    
    
    df.reset_index(drop=True, inplace=True)    

    # split train test data by whole slide or bbox
    train_inds, test_inds = paps_data_split(df, args.ratio, args.seed)

    train = df.iloc[train_inds]
    test = df.iloc[test_inds] 
    
    train.reset_index(drop=True, inplace=True)  
    test.reset_index(drop=True, inplace=True)      
    
    # need to check ratio 0.25, if not change seed
    print('train {}, test {} for bbox'.format(len(train_inds), len(test_inds)))
    print('train {}, test {} for wsi'.format(len(train.task.unique()),
                                             len(test.task.unique())))
    print('need to check train test ratio , if not good, change seed')
    
    train.to_csv(args.saved_path + 'train.csv', index=None)
    test.to_csv(args.saved_path + 'test.csv', index=None)
    
    print('csv for train/test files was made')
