#!/usr/bin/python

import numpy as np
import math
import random

from nn import *

parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
#parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of epoch times')
parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
args = parser.parse_args()

# define optimizer
optimizer = SGD(lr=args.lr)

# build a neural network
model = MLP()
model.add_layer(Affine(784,1000))
model.add_layer(ReLU())
model.add_layer(Affine(1000,1000))
model.add_layer(ReLU())
model.add_layer(Affine(1000,10))

optimizer.setup(model)

train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst = train(model,optimizer,epoch=args.epoch,batchsize=args.batchsize)
