#!/usr/bin/env python
# coding: utf-8

# # Pre-train model - 2

# ###  pretrain mobilenet_v2/classification model from tensorflow_hub as hub


import numpy as np
import time
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import load_img,img_to_array

import datetime


# pretrain models

#mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"



IMAGE_SHAPE = (224, 224)

# load model
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])

# show model summary
classifier.summary()


# #### Image -1

# load images
uniform = load_img('colllege-uniform.jpeg' , target_size=(224,224,3))
uniform = uniform.resize(IMAGE_SHAPE)
uniform


# reshape & normalize images
uniform = np.array(uniform)/255.0
print(uniform.shape)
uniform

# reshape images new way
uniform[np.newaxis, ...].shape


# Predict the image from 1001 features
result = classifier.predict(uniform[np.newaxis, ...])
print(result)   # under the array
result.shape

# print result
result[0].argmax()

# check labels for prediction
predicted_class = tf.math.argmax(result[0], axis=-1)
predicted_class


# Download the ImageNet labels file if not already present.
# We use tf.keras.utils.get_file to download the file from a URL if it's not already present in the cache.

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# 'ImageNetLabels.txt',   The name to give the file
# The URL where the file is located

labels_path



# Read the ImageNet labels file and convert it into a NumPy array.
# The file contains labels, each on a separate line. These are split into individual strings and stored as an array.

imagenet_labels = np.array(open(labels_path).read().splitlines())
imagenet_labels


# find prediction index for get label
imagenet_labels[870]



# show predicted image

plt.imshow(uniform)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())



# #### Image -2
# load images
car = load_img('car.jpeg', target_size=(224,224,3))
car


# reshape & normalize images
car = np.array(car)/255.0
print(car.shape)
car


# reshape images new way
car[np.newaxis, ...].shape


# Predict the image from 1001 features
result2 = classifier.predict(car[np.newaxis, ...])
print(result2)   # under the array
result2.shape


# print result2
result2[0].argmax()



# check labels for prediction
predicted_class2 = tf.math.argmax(result2[0], axis=-1)
predicted_class2


# Read the ImageNet labels file and convert it into a NumPy array.
# The file contains labels, each on a separate line. These are split into individual strings and stored as an array.

imagenet_labels1 = np.array(open(labels_path).read().splitlines())
imagenet_labels1

# find prediction index for get label
imagenet_labels1[818]


# show predicted image
plt.imshow(car)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class2]
_ = plt.title("Prediction: " + predicted_class_name.title())
