#!/usr/bin/env python
# coding: utf-8

# ##  MultiClass Perceptron with Encoders
# import the library

import numpy as np
import pandas as pd 
import  matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# create the dataset
X,y = make_classification(n_samples=1000,n_features=4,noise=30 ,random_state=0)
X = dataset.iloc[:,0:-1].values
print(X)

y = dataset.iloc[:,[-1]].values
print(y)


# import the One Hot Encoder for convert text to binary
from sklearn.preprocessing import OneHotEncoder

OHE = OneHotEncoder(sparse=False)

# sparse used in LabelEncoder
# transform the dependent var

y = OHE.fit_transform(y)
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(10,activation = tf.nn.relu, input_dim=4,name ='HiddenLayer1'))
model.add(Dense(10,activation = tf.nn.relu,name = 'HiddenLayer2'))
model.add(Dense(10,activation = tf.nn.relu,name = 'HiddenLayer3'))
model.add(Dense(3,activation = tf.nn.softmax,name = 'OutputLayer'))
model.summary()


# compile Model

# model.compile(loss=tf.losses.categorical_crossentropy,optimizer=tf.optimizers.Adam(),metrics=["aacuracy"])

model.compile(loss=tf.losses.categorical_crossentropy,optimizer=tf.optimizers.Adam(),metrics=["accuracy"])

# train model
model.fit(X_train,y_train,epochs=100)


# evaluate the model 
model.evaluate(X_test,y_test)


# predict model
y_pred = model.predict(X_test)
print(y_pred)
y_pred.round(2)


# fing higher values
y_pred = np.argmax(y_pred,axis=-1)
print(y_pred)

y_test = np.argmax(y_test,axis=-1)
y_test

# import the evaluaation matrics
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
