#!/usr/bin/python

import argparse

import numpy as np
import math
import random

import urllib.request
import glob

import gzip
import pickle

from nn import *
from preprocess import *

def train(model,optimizer,epoch,batchsize):

    for epo in range(epoch):
        sum_loss = 0
        pred_y = []

        # 訓練データをランダムに並び替える
        random_sort = np.random.permulation(train_data)

        for i in range(0,len(train_data),batchsize):
            x = train_data[random_sort[i:i+batchsize]]
            t = train_label[random_sort[i:i+batchsize]]

            loss = model.forward(x,t)
            model.backward()
            optimizer.update()

            sum_loss += loss

            pred = np.argmax(model.y,axis=1).tolist()
            pred_y.extend(pred)
        
        train_loss = sum_loss / len(train_data)
        train_accuracy = np.sum(one_hot(pred)*train_label[random_sort])/len(train_data)

        sum_loss = 0
        pred_y = []
        for i in range(0, len(test_data), batchsize):
            x = test_x[i: i+batchsize]
            t = test_y[i: i+batchsize]

            sum_loss += model.forward(x, t, train_config=False)
            pred = np.argmax(model.y, axis=1).tolist()
            pred_y.extend(pred)
        
        test_loss = sum_loss / len(test_data)
        test_accuracy = np.sum(one_hot(pred)*test_label[random_sort])/len(test_data)

        print('train | loss {:.4f},accuracy{:.4f}'.format(float(train_loss),train_accuracy))
        print('test | loss {:.4f},accuracy{:.4f}'.format(float(test_loss),test_accuracy))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    #parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
    parser.add_argument('--epoch', '-e', type=int, default=14, help='Number of epoch times')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    args = parser.parse_args()

    # -*- モデルの定義 -*-

    # optimizerを定義
    optimizer = SGD(lr=args.lr)

    # ニューラルネットワークの構成を定義
    model = MLP()
    model.add_layer(Affine(784,1000))
    model.add_layer(ReLU())
    model.add_layer(Affine(1000,1000))
    model.add_layer(ReLU())
    model.add_layer(Affine(1000,10))

    optimizer.setup(model)

    # -*- MNISTのデータをzipファイルとして指定urlからダウンロードし前処理をする -*-

    # データのあるURL
    url = 'http://yann.lecun.com/exdb/mnist/'

    # 保存先ディレクトリ
    dataset_dir = 'data'

    # ダウンロードするファイル
    key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']

    if len(glob.glob('data/*'))!=0:
        print("files are already downloaded")
    else:
        print("download dataset")
        for v in key_file.values():
            file_path = dataset_dir + '/' + v
            urllib.request.urlretrieve(url+v,file_path)
        print("complete")

    # numpyの配列に変換
    print("convert to numpy.ndarray")
    train_data,train_label = load_image_and_label(files[0],files[1])
    test_data,test_label = load_image_and_label(files[2],files[3])
    print("complete")

    #print(train_data.flags)

    # 学習データにノイズを加える
    print("add noise")
    train_data_with_noise = np.zeros((len(train_data),len(train_data[0])))
    for i in range(len(train_data)):
        for j in range(len(train_data[0])):
            p = random.randint(1,100)
            if p>=5:
                q = random.randint(0,255)
                train_data_with_noise[i][j] = q
            else:
                train_data_with_noise[i][j] = train_data[i][j]
    print("finished")

    """
    noise(train_data,0)

    # データの正規化
    train_data = normalize(train_data)
    test_data = normalize(test_set['data'])

    # ラベルをone-hot表現に変換
    train_label = one_hot(train_set['labels'])
    test_label = one_hot(test_set['labels'])

    # 学習させる
    #train(model,optimizer,epoch=args.epoch,batchsize=args.batchsize)
    """
