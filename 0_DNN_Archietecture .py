#!/usr/bin/env python
# coding: utf-8

# ###  Structure of DNNs
# ##### A Deep Neural Nnetwork typically includes:
# ##### Input Layer: The first layer that receives raw data (e.g., images, text).
# 
# ##### Hidden Layers: Multiple layers between the input and output layers that transform data and detect intricate patterns. DNNs can have hundreds or even thousands of hidden layers, making them capable of learning very complex representations.
# 
# ##### Output Layer: Produces the final prediction based on the processed data.The depth of a DNN refers to the number of hidden layers it contains. The deeper the network, the more complex features it can learn from the input data.

# ## practicle Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Load the dataset

dataset = pd.read_csv('sat.csv')

print(dataset)


# plot the distribution

plt.scatter(dataset['SAT'],dataset['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


X = dataset.iloc[:,0].values
print(X)

y = dataset.iloc[:,1].values
print(y)


# import Libraries

import keras 
from keras.models import Sequential
from keras.layers import Dense

# dense : specify Neuron 
#Sequential: This is a type of model in Keras that allows you to build a neural network layer by layer in a linear stack.

# Build the model
model = Sequential([
    Dense(10, activation="relu", input_shape=[1], 
    kernel_regularizer=keras.regularizers.l2(0.02), 
    kernel_initializer="he_normal"),
    Dense(10,activation="relu"),
    Dense(10,activation="relu"),
    Dense(1)
])

# format 10*10 neuron
# activation="relu" (for regression)
# kernel_regularizer=keras.regularizers.l2  (Used for add error in case of Overfitting)
# kernel_initializer = Weight Initializer

# summarize the model
model.summary()


# Used loss function : measure the accuracy of prediction

model.compile(loss="mean_squared_error",optimizer="adam")


# Adam Optimization : the Adam optimizer is a powerful tool that enhances the training process of deep learning models 
#  by providing adaptive learning rates, speeding up convergence, and maintaining robustness across various applications.


# Train model

model.fit(X,y,epochs=100)
# epochs : number of iteration

