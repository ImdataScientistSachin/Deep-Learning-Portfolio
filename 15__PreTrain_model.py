#!/usr/bin/env python
# coding: utf-8

# # Pre-train model - 1

# ### Use keras VGG16(weights='imagenet') pretrain model to predict

# import the library
import tensorflow as tf


# Pre- Trained model libraries
from keras.models import Model             # Used for creating and manipulating models
from keras.applications.vgg16 import VGG16  

# Pre-trained VGG16 model for image classification
from keras.utils import load_img,img_to_array,array_to_img
import numpy as np
import matplotlib.pyplot as plt


model = VGG16(weights='imagenet')

# weights='imagenet': Loads weights trained on ImageNet, 
# a large dataset commonly used for benchmarking image classification models.
"""
tf.keras.applications.VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
"""
# include_top=True: This parameter indicates whether to include the fully connected layers (the "top" of the model)
# that are used for classification. If set to True, the model will include these layers; if False, it will only include the convolutional base.
# seem models summary

model.summary()


# load some images 
img = load_img('horse1.jpg', target_size=(224,224,3))
img


#("conver", "images", "to", "array", "format")

img = img_to_array(img)
img


# resizing images
img = img.reshape(1,224,224,3)
img.shape


# check prediction
pred = model.predict(img).round(3)
pred


# 1 of 1000 features
pred.shape

# get the index of the highest probability
np.argmax(pred)


from keras.applications.vgg16 import decode_predictions


# The decode_predictions function in Keras is used to decode the predictions from a pre-trained model, 
# such as VGG16, trained on ImageNet. 
#  It converts the model's output probabilities into human-readable class labels and scores.

print('Predicted:', decode_predictions(pred, top=3)[0])

# Decode the predictions into human-readable labels and scores
# The top=3 parameter specifies that we want the top 3 predictions
# Use preds instead of pred to avoid NameError


# prediction 2 .
img_2 =  load_img('chair.jpeg',target_size = (224,224,3))
img_2


# image to array
img_2 = img_to_array(img_2)

# chack shape
print(img_2.shape)
# reshape it
img_2 = img_2.reshape(1,224,224,3)
img_2


# prediction 2
pred2 = model.predict(img_2)
pred2.round(3).argmax()


from keras.applications.vgg16 import decode_predictions
print('Predicted:', decode_predictions(pred2, top=3)[0])
# Decode the predictions into human-readable labels and scores
# The top=3 parameter specifies that we want the top 3 predictions