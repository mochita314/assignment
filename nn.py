#!/usr/bin/python

import numpy as np
import math
import random

# Implement activation function layers

class Sigmoid():

    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    # backword propagation
    def backward(self,dout):
        dout = dout*self.y*(1-self.y)
        return dout

class ReLU():

    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = x*(x>0)
        self.y = y
        return y
    
    # backward propagation
    def backward(self,dout):
        dout = dout*(self.x>0)
        return dout

class Softmax():

    # the way of implementation of backward propagation is different from other layers
    # as this is the last layer

    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
        self.param = False
    
    def __call__(self,x):
        self.x = x
        exp_x = np.exp(x-x.max(axis=1,keepdims=True)) # prevent overflow
        sum_exp_x = np.sum(exp_x,axis=1,keepdims=True)
        y = exp_x / sum_exp_x
        self.y = y
        return y
    
    def backward(self,dout=1):
        dout = (self.y-self.t) / len(self.x)
        return dout

# Implement linear layer
class Affine():

    def __init__(self,input_dim,output_dim):
        self.x = None
        self.y = None
        self.param = True
        self.params = {}
        # initialize weight matrix
        std = np.sqrt(2.0 / input_dim)
        self.params['W'] = std * np.random.randn(input_dim,output_dim)
        # initialize bias vector
        self.params['b'] = np.zeros(output_dim)
        self.grads = {}
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = np.dot(x,self.params['W']) + self.params['b']
        self.y = y
        return y
    
    # backward propagation
    def backward(self,dout):
        dout = np.dot(dout,self.params['W'].T)
        self.grads['dW'] = np.dot(self.x.T,dout)
        self.grads['db'] = np.sum(dout,axis=0)
        return dout

class SGD():

    def __init__(self,lr):
        self.lr = lr
        self.network = None
    
    def setup(self,network):
        self.network = network
    
    def update(self):
        for layer in self.network.layers:
            if layer.param: # don't apply for actibavtion function layers
                layer.params['W'] -= self.lr * layer.grads['dW']
                layer.params['b'] -= self.lr * layer.grads['db']

# Implement Multilayer perceptron
class MLP():

    def __init__(self,layers=[]):
        self.layers = layers
        self.t = None
    
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def forward(self,x,t):
        self.t = t
        self.x = x
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + delta))
        return self.loss
    
    def backward(self):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

def train(model,optimizer,epoch,batchsize):

    return train_loss_lst,train_acc_lst,test_loss_lst,test_acc_lst