#!/usr/bin/python

from nn import *

parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

model = MLP()
