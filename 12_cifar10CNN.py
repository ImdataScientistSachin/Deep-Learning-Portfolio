#!/usr/bin/env python
# coding: utf-8

# ## Build Deep Learning  Model on Cifar10 dataset


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import the Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import datasets, layers, models
sns.set_style('whitegrid')


# load the dataset and split the dataset
(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()
train_images.shape
train_labels
train_labels.shape
test_images.shape
test_labels
train_images[0]
train_labels[0][0]
train_images[10]
train_labels[10]
train_labels[10][0]

# visualize the dataset
plt.imshow(train_images[0])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()



plt.imshow(train_images[0])
plt.title(train_labels[0][0])  # print labels
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()


plt.imshow(train_images[10])
plt.title(train_labels[10][0]) # print labels
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()



# create List with datasets classes 
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# print class names

plt.imshow(train_images[10])
plt.title(class_names[train_labels[10][0]])# print class name
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()


plt.imshow(train_images[101])
plt.title(class_names[train_labels[101][0]])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()


# plot multiple images with labels
plt.figure(figsize=(10,10))
for i in range(25):# print 25 images 
    plt.subplot(5,5,i+1) # 5*5
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary) # print test images
    plt.xlabel(class_names[test_labels[i][0]])  # print Labels
plt.show()

# normalize the images
train_images,test_images = train_images / 255.0, test_images / 255.0

# scale images
train_images[0] # print training image of 0 index
#  pixel value between 0 to 1

# prepare model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.summary()

#  ( 32*32)# 1channel = 1024
#  1024*3 3 channels = 3072


# Add layers to the model
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()



# compile the model
model.compile(optimizer='adam',
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])


# train the model
history = model.fit(train_images,train_labels,epochs=40,
                   validation_data=(test_images,test_labels))



# plot the distributin of trining_loss and validation_loss 

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0, 3])

plt.legend(loc=0)
plt.show()



# plot the distributin of trining_accuracy  and validation_accuracy 

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(loc=0)
plt.show()



# load i=one of test Images

plt.imshow(test_images[7000])
plt.title(class_names[test_labels[7000][0]])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()



# model Prediction 
model.predict(test_images[7000].reshape(1,32,32,3))
# 1 image,32*32 , 3 channel

# make prediction
model.predict(test_images[7000].reshape(1,32,32,3)).round(3)


# scaled prediction value 
np.argmax(model.predict(test_images[7000].reshape(1,32,32,3)))



# find labels 
class_names[np.argmax(model.predict(test_images[7000].reshape(1,32,32,3)))]

# model doesnt evaluate correctly

# Evaluate the model
model.evaluate(test_images,test_labels)
# test loss and test accuracy