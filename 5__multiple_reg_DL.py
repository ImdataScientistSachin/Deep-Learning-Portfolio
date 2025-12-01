#!/usr/bin/env python
# coding: utf-8

# ## Multiclass regretion in DeepLearning

# import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


from sklearn.datasets import make_regression

X,y = make_regression(n_samples=1000,n_features=6, noise=20,random_state=0)
X
y

# import the keras Library
from keras.models import Sequential
from keras.layers import Dense

model= Sequential()
model.add(Dense(20,activation='relu',input_dim=6,kernel_initializer='he_uniform'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))

model.summary()

import tensorflow as tf

optimizers = tf.keras.optimizers.Adam(learning_rate=0.0001)


# Compile the model
model.compile(loss='mse',optimizer=optimizers)
# define loss and Optimization function


# trian model 
model.fit(X,y,epochs=200)