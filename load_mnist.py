#!/usr/bin/python

import urllib.request
import numpy as np
import gzip
import pickle

# download mnist data as zip files

url = 'http://yann.lecun.com/exdb/mnist/'
dst_path = '~/3A/ロボットインテリジェンス/assignment'

files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']

for _file_ in files:
    file_path = dst_path + '/' + _file_
    urllib.request.urlretrieve(url+_file_,file_path)

# open zip files 