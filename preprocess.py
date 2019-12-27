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

def noise(arr1,d):

    arr2 = np.zeros((len(arr1),len(arr1[0])))
    if d==0:
        pass
    else:
        for i in range(len(arr1)):
            if (i+1)%1000 == 0:
                print('{:.2f} percent complete'.format((i+1)/600))
            for j in range(len(arr1[0])):
                p = random.randint(1,100)
                if p<=d:
                    q = random.randint(0,255)
                    arr2[i][j] = q
                else:
                    arr2[i][j] = arr1[i][j]
    
    arr1 = arr2
    return arr1

def normalize(arr):

    arr = arr.astype(np.float32)
    arr /= 255

    return arr

def one_hot(label):

    one_hot_label = np.zeros((label.size,10))
    for i in range((label.size)):
        one_hot_label[i][label[i]] = 1

    return one_hot_label

