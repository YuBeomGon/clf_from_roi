# this is for making json file one clase detection, train, test json file would be made

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


def set_negative(df) :
    df = df.copy()
    # object detection is one class, abnornal
    df = df[df['label_id'] != 'Negative']
    print('***label****', df.label_id.unique())  
    #df.label_id = df.label_id.apply(lambda x : 'Abnormal')
    df.loc[:,'label_id'] = 'Abnormal'
    print(df.label_id.unique())  
    df.reset_index(drop=True, inplace=True)   
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    
    df = pd.read_csv(args.saved_path + 'df.csv')
    org_size = df.shape

    # label id is for detection, one class detecter, abnormal or not
    # further, use more class for detector, 
    # and change to abnormal or int in inference, may got more performance    

    df['label_id'] = df.label.apply(lambda x : CLASS_MAPPER[str(x)])
    df = drop_wrong(df, columns='label_id')
    
    # take abnormal only,
    # df['label_id'] = df.label_id.apply(lambda x : 'negative' if 'Benign' in x or 'Negative' in x else 'abnormal')
    normal_size = df.shape
    
    print('org_size {} normal size {} '.format(org_size, normal_size))   

    # split train test data by whole slide or bbox
    train_inds, test_inds = paps_data_split(df, args.ratio, args.seed, columns='label_id')

    train = df.iloc[train_inds]
    test = df.iloc[test_inds] 

    train = set_negative(train)
    test = set_negative(test)
    
    # need to check ratio 0.25, if not change seed
    print('train {}, test {} for bbox'.format(len(train_inds), len(test_inds)))
    print('train {}, test {} for wsi'.format(len(train.task.unique()),
                                             len(test.task.unique())))
    print('need to check train test ratio , if not good, change seed')
    
    
    # for change df to json file for coco evaluation
    # usually no need to add background class, detection model add internally
    # but need to check by detection model
    # cats_dic={'Candida':0,'ASC-US':1, 'LSIL':2, 'HSIL':3, 'Negative':4,
    #           'Carcinoma':5, 'ASC-H':6, 'Benign':7, 'background':8}    
    cats_dic={'Abnormal':1}    
    
    convert(cats_dic, train, json_file=args.saved_path + 'train.json')
    convert(cats_dic, test, json_file=args.saved_path + 'test.json')
    
    print('json file was made')
