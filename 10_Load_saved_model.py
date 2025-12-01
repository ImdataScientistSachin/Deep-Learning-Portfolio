#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow.keras.utils import load_img,img_to_array
import tensorflow as tf



# saved model 
"""
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the file path where the model will be saved
filepath = 'my_best_flower.h5'

# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    # Specify the file path
    filepath=filepath,
    # Monitor the validation loss
    monitor='val_loss',
    # Print messages when saving the model
    verbose=1,
    # Only save the model with the best validation loss
    save_best_only=True,
    # Save the model with the minimum validation loss
    mode='min'
)


"""

model = tf.keras.models.load_model('my_best_flower.h5')
model.summary()

# test the model 
test = load_img('rose.jpeg',target_size=(180,180))
test
test = img_to_array(test)
test = test/255.0
test.round()

# reshape 
test = test.reshape(1,180,180,3)
test.shape

predictions = model.predict(test)
print(predictions)


score = tf.nn.softmax(predictions[0])
score = np.array(score).round()
score

# link of datasets
# https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip
