#!/usr/bin/python

import argparse

import matplotlib.pyplot as plt
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

    tr_loss_lst = []
    tr_accuracy_lst = []
    te_loss_lst = []
    te_accuracy_lst = []

    for epo in range(epoch):
        sum_loss = 0
        pred_y = []

        # 訓練データをランダムに並び替える
        random_sort = np.random.permutation(len(train_data))

        print('start training')
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
        train_accuracy = np.sum(np.eye(10)[pred_y]*train_label[random_sort])/len(train_data)

        tr_loss_lst.append(train_loss)
        tr_accuracy_lst.append(train_accuracy)

        print('finished')

        print('start test')

        sum_loss = 0
        pred_y = []
        for i in range(0, len(test_data), batchsize):
            x = test_data[i: i+batchsize]
            t = test_label[i: i+batchsize]

            sum_loss += model.forward(x,t)
            pred = np.argmax(model.y, axis=1).tolist()
            pred_y.extend(pred)
        
        test_loss = sum_loss / len(test_data)
        test_accuracy = np.sum(np.eye(10)[pred_y]*test_label)/len(test_data)

        te_loss_lst.append(test_loss)
        te_accuracy_lst.append(test_accuracy)

        print('finished')

        print('train | loss {:.4f}, accuracy {:.4f}'.format(float(train_loss),train_accuracy))
        print('test | loss {:.4f}, accuracy {:.4f}'.format(float(test_loss),test_accuracy))

    return tr_loss_lst,tr_accuracy_lst,te_loss_lst,te_accuracy_lst

def show_result(lst1,lst2):
    plt.plot(np.arange(len(lst1)), np.asarray(lst1), label='train')
    plt.plot(np.arange(len(lst2)), np.asarray(lst2), label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    #parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
    parser.add_argument('--size', type=int, default=1000, help='Number of neurons in the middle layer')
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoch times')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    args = parser.parse_args()

    # -*- モデルの定義 -*-

    # optimizerを定義
    optimizer = SGD(lr=0.5)

    # ニューラルネットワークの構成を定義

    #model = MLP([Affine(784,1000),Sigmoid(),Affine(1000,1000),Sigmoid(),Affine(1000,10),Softmax()])

    model = MLP([])
    size = args.size
    model.add_layer(Affine(784,size))
    model.add_layer(Sigmoid())
    model.add_layer(Affine(size,size))
    model.add_layer(Sigmoid())
    model.add_layer(Affine(size,10))
    model.add_layer(Softmax())

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

    # print(train_data.flags)

    # 学習データにノイズを加える
    print("add noise")
    train_data = noise(train_data,0)
    print("finished")

    # データの正規化
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    # ラベルをone-hot表現に変換
    train_label = one_hot(train_label)
    test_label = one_hot(test_label)

    # for debug
    # print(train_data[:10])
    # print(train_label[:10])
    # 学習させる

    lst1,lst2,lst3,lst4 = train(model,optimizer,epoch=args.epoch,batchsize=args.batchsize)

    show_result(lst1,lst3)
    show_result(lst2,lst4)
