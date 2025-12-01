#!/usr/bin/env python
# coding: utf-8

# #  MNIST

# ##### The MNIST database (Modified National Institute of Standards and Technology database) is a widely used collection of handwritten digits commonly used for training image processing systems and machine learning models.
# 
# ##### Key Features:
# 
# ##### The MNIST dataset comprises 60,000 training images and 10,000 testing images.
# ##### The images are grayscale and have a size of 28x28 pixels2.
# ##### The digits are normalized to fit within a 28x28 pixel bounding box and anti-aliased, which introduces grayscale levels

# ### Practicle Implementation

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
sns.set_style('whitegrid')

# load the dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train.shape


# find the shape of dataset 
type(X_train)
X_train[0]
X_test.shape

# labelled dataset
y_train


# plot the distributin
plt.imshow(X_train[0],cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(X_train[56],cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()


plt.imshow(X_train[0],cmap='gray_r')
plt.title(y_train[0])    # show the labelled value
plt.xticks([])
plt.yticks([])
plt.show()


plt.imshow(X_train[40],cmap='gray_r')
plt.title(y_train[40])
plt.xticks([])
plt.yticks([])
plt.show()


# plot the subplot for multiple datasets
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i],cmap='gray_r')
    plt.title(y_train[i])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()    
plt.show()

# print i
for i in range(9):
    print(i)


# Normalize & casting
X_train,X_test = X_train/255.0, X_test/255.0
# 255 is the highest value in the dataset
X_train[0]


# plot the Normalize dataset
plt.imshow(X_train[0],cmap='gray_r')
plt.show


# prepare model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()


# Compile model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# train the model
model.fit(X_train,y_train,epochs=10)
model.evaluate(X_test,y_test)


# plot the distribution
plt.imshow(X_test[60],cmap='gray_r')
plt.title(y_test[60])
plt.xticks([])
plt.yticks([])
plt.show()


# model prediction
model.predict(X_test[60].reshape(1,28,28))


# prediction with round based on index
model.predict(X_test[60].reshape(1,28,28)).round(3)


# reshaping the dataset predicted value
np.argmax(model.predict(X_test[60].reshape(1,28,28)))
X_test[6].shape
X_test[6].reshape(1,28,28)