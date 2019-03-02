#!/usr/bin/python
import numpy as np 
import pandas as pd 
import lightgbm as lgb
import json
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import tensorflow as tf
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)

def dataAbstruct(jsonPath, segment):
    with open(jsonPath) as json_data:
        data = json.load(json_data)
        print(data.keys())
    for key, value in data.items():     
        if key == segment:
            print('Abstruct: ',key)
            return value


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tranJsonPath = './jinnan2_round1_train_20190222/train_no_poly.json'
    images = {}
    images = dataAbstruct(tranJsonPath, 'images')
    annotations = dataAbstruct(tranJsonPath, 'annotations')
    for data in images:     
        image = ('./jinnan2_round1_train_20190222/restricted/' + data['file_name'])
        # 读入二进制文件
        image_raw = tf.read_file(image)
 
        # 解码为tf中的图像格式
        img = tf.image.decode_jpeg(image_raw, channels=3)  #Tensor
        with tf.Session() as sess:
            img_ = img.eval()
            #print(img_.shape)
        for annotation in annotations:
            if data['id'] == annotation['id']:
                bbox = annotation['bbox']
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                imageShow = cv2.imread(image)
                cv2.rectangle(imageShow, p1, p2, (255,0,0), 2, 1)
                cv2.imshow('image', imageShow)
                c = cv2.waitKey(0)
                if c & 0xFF == ord('q'):
                    break        
 
        #plt.figure(1)
        #plt.imshow(img_)
        #plt.show()

    '''
    with open('./jinnan2_round1_submit_20190222.json') as json_data:
        data = json.load(json_data)
    train = pd.DataFrame.from_dict(data)  
    #test  = pd.read_json('datalab/7955/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
    print(train)
    stats = []
    for col in train.columns:
        #print(col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype)
        stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))
    
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=False)[:10]
    target_col = "收率"
    #plt.figure(figsize=(8,6))
    #plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
    #plt.xlabel('index', fontsize=12)
    #plt.ylabel('yield', fontsize=12)
    #plt.show()
    '''
