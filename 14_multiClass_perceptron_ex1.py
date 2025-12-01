#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# load dataset

from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000,n_features=6,n_classes=2,random_state=0)

X = X.round()

print(X)


y=y.round()

print(y)


# prepare dataset for train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# prepare the model

model = Sequential()
model.add(Dense(10,activation = 'relu', input_dim =6))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation= 'sigmoid')) # output layer depends on output classes
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# compile the model
