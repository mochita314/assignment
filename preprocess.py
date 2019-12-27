#!/usr/bin/python

# -*- データの前処理を行う関数を定義したファイル -*-

import urllib.request
import numpy as np
import gzip
import pickle
import random

def load_image_and_label(data_file_name,label_file_name):

    dic = {}
    
    dataset_dir = 'data'

    file_path = dataset_dir + '/' + data_file_name
    with gzip.open(file_path,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=16)
    data = data.reshape(-1,784)
    
    file_path = dataset_dir + '/' + label_file_name
    with gzip.open(file_path,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset=8)
    label = labels

    return [data,label]

def noise(arr,d):

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            p = random.randint(1,100)
            if p>=d:
                q = random.randint(0,255)
                arr[i][j] = q

    return arr

def normalize(arr):

    arr = arr.astype(np.float32)
    arr /= 255

    return arr

def one_hot(label):

    one_hot_label = np.zeros((label.size,10))
    for i in range((label.size)):
        one_hot_label[i][label[i]] = 1

    return one_hot_label

