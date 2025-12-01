#!/usr/bin/env python
# coding: utf-8

# ## Multiclass classification in Deep Learning

# Import the Libraires

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# import the dataset
dataset = pd.read_csv('iris.csv')
dataset


# prepare the dataset
X = dataset.iloc[:,0:-1].values
X


y = dataset.iloc[:,-1].values
y


# transform the data using label Encoder (text to class)

# import the library 

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()


y = LE.fit_transform(y)
y


# prepare dataset for train and test
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape


from keras.models import Sequential
from keras.layers import Dense


# prepare the model

model = Sequential()
model.add(Dense(10,activation = 'relu', input_dim = 4))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation= 'softmax')) # output layer depends on output classes
model.summary()

# compile the model 
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics =["accuracy"])

# in multi class we use sparse.categorical_crossentropy for LabelEncoder

# trian model 
model.fit(X_train,y_train,epochs=200)


# evaluate the model
model.evaluate(X_test,y_test)

# model prediction
y_pred = model.predict(X_test)
y_pred.round()

y_pred = np.argmax(y_pred,axis=-1)
y_pred


# compare with actual value
y_test


# print Evaluation matrix for compare both actual and predicted value
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))