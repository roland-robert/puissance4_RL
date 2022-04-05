import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from collections import deque
import random as rd
from puissance4 import Game
import matplotlib.pyplot as plt
import time as time

import psutil as psutil #cpu usage
import os as os

from MCTS import MCTS

import pickle as pickle #charger le training data

class Hyperparameters:
    def __init__(self):
        
        self.lr = 0.01
        
        #initializer for Gaussienne
        self.seed = 10
        self.std = 0.1
        self.mean = 0
        
        #loss, initializer weight
        #self.loss = tf.keras.losses.MeanSquaredError()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metric=tf.keras.metrics.MeanSquaredError()
        self.kern_initializer = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.std, seed=self.seed)
        #self.kern_initializer = tf.keras.initializers.GlorotUniform(seed = self.seed)
        self.bias_initializer = tf.keras.initializers.Zeros()

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

class Policynet:
    def __init__(self, h=6, w=7, model = 0) -> None:
        self.h = h
        self.w = w
        self.hpp = Hyperparameters()
        self.game = Game(h,w) #just for rules
        if model == 0:
            
            self.model = Sequential()
            self.model.add(layers.Conv2D(512, (4, 4), activation="elu", 
                                         input_shape=(h, w, 3),
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(units = 512, activation = 'tanh', 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Dense(units = 512, activation = 'elu', 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Dense(units = w, activation = 'softmax',
                                        kernel_initializer=self.hpp.kern_initializer,
                                        bias_initializer=self.hpp.bias_initializer))

            self.model.compile(optimizer = self.hpp.optimizer, 
                   loss ='categorical_crossentropy',
                   metrics=[tf.keras.metrics.MeanSquaredError()])
                   #'categorical_crossentropy'
            print(self.model.summary())
            #Adam, SGD
        else:
            self.model = model
        self.epoch = 1 #number of games played

    def get_probabilities_adjusted(self, X): #récupère les probabilités
        #ajusted : proba des coups valides ajustés, proba des coups invalides inchangés
        board = self.game.convert_state_to_board_bitmap(X)
        available_moves = self.game.get_valid_moves(board)
        
        L = []
        L.append(X)
        Y = self.model.predict(np.array(L))[0]
        valid_probabilities_sum = sum([Y[i] for i in available_moves])
        for i in available_moves:
            Y[i] = Y[i]/valid_probabilities_sum
            
        return Y,valid_probabilities_sum
        
    def get_probabilities(self, X): #récupère les probabilités = prediction
        #(prédiction brute, sans ajustements)
        board = self.game.convert_state_to_board_bitmap(X)
        available_moves = self.game.get_valid_moves(board)
        L = []
        L.append(X)
        Y = self.model.predict(np.array(L))[0]
        return Y
    def get_move(self, X, training = False): #DO NOT USE :o
        board = self.game.convert_state_to_board_bitmap(X)
        available_moves = self.game.get_valid_moves(board)
        if training: #entrainant => randomness
            r = self.hpp.real_randomness(self.epoch)
            rnd = rd.random()
            if rnd<r:
                return rd.choice(available_moves)
        Y = self.get_qs(X)
        if self.hpp.allow_illegal_moves and training == True:
            return np.argmax(Y)
        
        themax = -np.inf
        imax = -1
        for i in available_moves:
            if Y[i]>themax:
                themax = Y[i]
                imax = i
        if imax == -1:
            print("PROBLEME !")
            
            print("Board : ", board)
            print("Available moves : ", available_moves)
            print("qs : ", Y)
        return imax


class Valuenet:
    def __init__(self, h=6, w=7, model = 0) -> None:
        self.h = h
        self.w = w
        self.hpp = Hyperparameters()
        self.game = Game(h,w) #just for rules
        if model == 0:
            
            self.model = Sequential()
            self.model.add(layers.Conv2D(512, (4, 4), activation="selu", 
                                         input_shape=(h, w, 3),
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(units = 512, activation = 'tanh', 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))
            self.model.add(layers.Dense(units = 512, activation = "elu", 
                  kernel_initializer=self.hpp.kern_initializer,
                  bias_initializer=self.hpp.bias_initializer))

            self.model.add(layers.Dense(units = 1, activation = 'tanh',
                                        kernel_initializer=self.hpp.kern_initializer,
                                        bias_initializer=self.hpp.bias_initializer))

            self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.hpp.lr), 
                   loss = tf.keras.losses.MeanAbsoluteError(),
                   metrics=[tf.keras.metrics.MeanSquaredError()])
            print(self.model.summary())
            #Adam, SGD
        else:
            self.model = model
        #params
        self.epoch = 1 #number of games played
    def get_value(self, X): #récupère les probabilités
        #
        board = self.game.convert_state_to_board_bitmap(X)
        available_moves = self.game.get_valid_moves(board)
        L = []
        L.append(X)
        Y = self.model.predict(np.array(L))[0] #liste de 1 nombres
        return Y[0]

class Trainer:
    def __init__(self,h =6, w = 7,  modelP = 0, modelV = 0):
        self.policynet = Policynet(h,w,modelP)
        self.valuenet = Valuenet(h,w,modelV)

    def get_value(self, state):
        val = self.valuenet.get_value(state)
        return val
    def get_policy(self, state):
        policy = self.policynet.get_probabilities(state)
        return policy