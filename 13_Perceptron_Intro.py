#!/usr/bin/env python
# coding: utf-8

# # Perceptron 
# 

# ### The perceptron is a fundamental concept in machine learning and deep learning, serving as the building block for more complex neural networks. It is primarily used as a binary classifier, which means it can categorize input data into one of two classes.
# 
# ##### Formula =   f(x)=h(w⋅x+b)
# 
# #### where:
#  ##### h is the activation function, w is the weight vector,
# #####  b is the bias term, which allows the model to shift the decision boundary away from the  origin.
# 
# ####  Key Components of a Perceptron: 
# ##### Input Layer: This consists of input neurons that receive data.
# ##### Weights: Each input has an associated weight that signifies its importance in the decision-making process.
# ##### Bias: A constant added to the weighted sum to adjust the output independently of the input values.
# ##### Activation Function: Determines if the neuron should "fire" or produce an output based on the weighted sum. Common functions include step functions, sigmoid.
# 
# 
# #### Types of Perceptrons:
# ##### Perceptrons can be categorized into two main types:
# ##### Single-Layer Perceptron: This model consists of only one layer of output nodes connected directly to input features, suitable for linearly separable data.
# ##### Multi-Layer Perceptron (MLP): This extends the single-layer perceptron by adding one or more hidden layers, enabling it to learn more complex patterns and handle non-linear classification problems ReLUorigin .
# 

# ### Practicle Implementation 


# Linear Regression (y= B0+b1*x)



import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')

# prepare the database

X = np.array([-1,0,1,2,3,4])

X = X.reshape(-1,1)

y = np.array([-3,-1,1,3,5,7])


# plot the distrribution

plt.scatter(X,y)
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,y)

model.score(X,y)

## finding the B0

model.intercept_

# finding the B1

model.coef_


# ## Using Perceptron

# import deep Learning Libraries(keras)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([Dense(units=1,input_shape=[1])])


# ### explanetion of  line 
# 
# ##### Sequential: This is a type of model in Keras that allows you to build a neural network layer by layer in a linear stack.It is particularly useful for simple architectures where each layer has exactly one input tensor and one output tensor.
# #####  Dense: This refers to a fully connected layer (also known as a dense layer). In this layer, every neuron receives input from all neurons in the previous layer.
# #####  units=1: This parameter specifies the number of neurons (or units) in this dense layer. Here, it indicates that there is only one neuron in the output layer.
# #####  input_shape=: This defines the shape of the input data that the model will accept. In this case, it indicates that each input sample will have one feature (a single-dimensional input).


# check Evaluation of model 

model.summary()


# # explanetion 
# 
# ##### Model: "sequential_1": This indicates the name of the model instance.
# ##### Layer (type): This column lists the layers in the model along with their types. Here, there is one layer:
# ##### dense_1 (Dense): This indicates that the layer is a dense (fully connected) layer named dense_1.
# ##### Output Shape: This column shows the shape of the output produced by each layer.
# ##### (None, 1): The None indicates that the batch size can vary (it's flexible), and 1 indicates that the output of this layer will have one feature. In a regression context, this means the model outputs a single continuous value.
# 
# ##### Param #: This column displays the number of parameters in each layer.
# 
# ##### Total params: 2: This line summarizes the total number of parameters in the entire model, which is 2 in this case.
# 
# ##### Each input feature has an associated weight, and there is also a bias term. Since there is one input feature and one output neuron, the calculation is:
# 
# 
#          Weights = 1 (input feature) × 1 (output neuron) = 1         
#           Bias = 1 (for the output neuron )
#          Total = 1 (weight) + 1 (bias) = 2 parameter
# ##### Trainable params: 2: This indicates that all parameters in this model are trainable, meaning they will be updated during training through backpropagation.


model.compile(loss="mean_squared_error",optimizer="sgd")



# #####  Use MSE loss function 
# ##### Usr SGD Optimizer : This defines the optimization algorithm to be used for updating the model's weights during training.updates weights incrementally based on each training example (or mini-batch)

#   train model

model.fit(X,y,epochs=500)

# 1.4420e-09: This notation represents a very small number (1.4420 × 10 power of -9)
# It indicates that the loss has decreased significantly, suggesting that the model has learned the underlying pattern in the data effectively.
0.000014420

# predict the Ouput

model.predict(np.array([4]))

# plot the distribution

plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()

# Evaluate the model 

model.evaluate(X,y)
# 1.2214e-09: This notation represents a very small number (1.2214 × 10 power of -9)
# It indicates that the loss is very low, suggesting that the model has learned the underlying pattern in the data effectively.

# get the weights and bias
model.get_weights()

# model.get_weights() is a method in Keras that retrieves 
# the current weights and biases of a model. 



# [array([[1.9999839]]= This array represents the weights of the dense layer in your model.

# array([-0.9999503]=This array represents the bias term associated with the output neuron.

#   y=(1.9965698×x)+(−0.9893654)

# So, the final linear equation learned by the model is approximately: