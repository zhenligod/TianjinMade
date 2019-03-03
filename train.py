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

def GetTrainPicList(input_dir):
    fileListArr = []
    for file in os.listdir(input_dir):
        temp_dir = os.path.join(input_dir, file)
        fileListArr.append(temp_dir)
    return fileListArr




if __name__ == '__main__':
    print(GetTrainPicList('./1'))