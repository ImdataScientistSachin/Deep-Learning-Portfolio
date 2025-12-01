#!/usr/bin/env python
# coding: utf-8

# # Implementation CNN on Fashion Mist dataset


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)



# import the Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from keras.datasets import mnist


# Load the dataaet

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print (train_images.shape)
train_images[0]
print (test_images.shape)
train_images[0].shape


# same as MNIST dataset
train_labels


# create a Variable to store labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize  the data
# plot the distribution

plt.imshow(train_images[0],cmap='gray_r')
plt.title(class_names[train_labels[0]])
plt.xticks([])
plt.yticks([])
plt.show()

# Print the label 
plt.imshow(train_images[70],cmap='gray_r')
plt.title(class_names[train_labels[70]])
plt.xticks([])
plt.yticks([])
plt.show()



# CNN 

#Format --  (num_of_img,width,height,clannel)

# Reshaping
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


print(train_images.shape)
print(train_images[0].shape)

# Visualize after reshaping
plt.imshow(train_images[77],cmap='gray_r')
plt.title(class_names[train_labels[77]])


# Casting
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

print(train_images[0])
print(test_images[0])


# Normalization 
train_images = train_images / 255.0
test_images = test_images / 255.0

print (train_images[0])
print (test_images[0])


#Apply  CNN
model = models.Sequential()
model.add(layers.Conv2D(28,(3,3),activation='relu',input_shape=(28,28,1))) #28-3+1=26
model.add(layers.MaxPooling2D((2,2))) # 13
model.add(layers.Conv2D(56,(3,3),activation='relu')) #13-3+1=11
model.add(layers.MaxPool2D((2,2))) #5
model.add(layers.Conv2D(56,(3,3),activation='relu')) #5-3+1=3
model.summary()


# Add Layers
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# trian the model
history = model.fit(train_images,train_labels,epochs=5,batch_size=10,validation_split=0.2)
history.history

# Visualize the training
#  Plot the distributin

epochs = 5
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc=0)
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc=0)
plt.title('Training and Validation Loss')
plt.show()


# Evaluate the model
model.evaluate(test_images,test_labels)

# Predictions
# check the data sample 

plt.imshow(test_images[2250],cmap='gray_r')
plt.title(class_names[test_labels[2250]])
plt.xticks([])
plt.yticks([])
plt.show()

# check the shape
test_images[2250].shape


# Test prediction
model.predict(test_images[1250].reshape(1,28,28))


# reshaping the predicted image 
model.predict(test_images[2333].reshape(1,28,28,1))
print (np.argmax(model.predict(test_images[1250].reshape(1,28,28))))

# check Prediction 2
class_names[np.argmax(model.predict(test_images[2250].reshape(1,28,28)))]
class_names[np.argmax(model.predict(test_images[2333].reshape(1,28,28,1)))]