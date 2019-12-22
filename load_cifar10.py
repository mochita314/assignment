#!/usr/bin/python

import urllib.request
import numpy as np
import os,sys
import pickle
from PIL import Image

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

dict = unpickle('cifar-10-batches-py/data_batch_1')