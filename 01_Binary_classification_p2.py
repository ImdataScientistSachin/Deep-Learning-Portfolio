#!/usr/bin/env python
# coding: utf-8

# ## Binary clsssifier in DeepLearning with matrics
# ##### precision, recall, f1-score , AUC (Area Under the ROC Curve)

# import the Libraries

import numpy as np
import pandas as pd


# import the dataset

dataset = pd.read_csv('binary_log.csv')
print(dataset)


# prepare the dataset 

dataset['Admitted'] = dataset['Admitted'].map({'Yes':1,'No':0})

dataset['Gender'] = dataset['Gender'].map({'Female':1,'Male':0})

print(dataset)

X = dataset.iloc[:,[0,2]].values
print(X)

y = dataset.iloc[:,1].values
print(y)

# import sklearn library

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# import the keras library

from keras.models import Sequential
from keras.layers import Dense


model = Sequential() # create function
model.add(Dense(10,activation='relu',input_dim=2,name = 'HiddenLayer1'))
model.add(Dense(10,activation='relu',name="HiddenLayer2"))
model.add(Dense(10,activation='relu',name='Hiddenlayer3'))
model.add(Dense(1,activation='sigmoid',name='OutputLayer'))

model.summary()

# compile the program with matrics
# precision, recall, f1-score , AUC (Area Under the ROC Curve)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.AUC()])

# ['accuracy']: In this case, you're asking the model to calculate and report the accuracy during training and testing. 


# train model
model.fit(X,y,epochs=100)

# evaluate Model
model.evaluate(X_test,y_test)

# prediction
y_pred = model.predict(X_test)
# print the predicted values
print(y_pred)

# conver the values into probability using round() 
y_pred = y_pred.round()
print(y_pred)


# print the Evaluation matrics
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# confusion matrix: It provides a summary of prediction results on a classification problem.
# classification report: It provides a detailed report on the precision, recall, F1-score, and support for each class in a classification problem.