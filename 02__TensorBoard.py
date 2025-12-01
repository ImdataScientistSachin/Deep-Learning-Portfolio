#!/usr/bin/env python
# coding: utf-8

# # TensorBoard

# ##### TensorBoard is an open-source visualization toolkit designed to help machine learning practitioners debug, optimize, and understand their models. It provides a suite of tools for tracking and visualizing metrics such as loss and accuracy, inspecting model graphs, viewing histograms of weights and biases, projecting embeddings into lower-dimensional spaces, and displaying data like images, text, or audio

# ### Key features include:
# 
# ##### Metrics Tracking: Monitor loss, accuracy, and other scalar values over time.
# 
# ##### Graph Visualization: Display computation graphs to understand model architecture.
# 
# ##### Histograms: Visualize changes in weights, biases, or other tensors across epochs.
# 
# ##### Embedding Projections: Reduce high-dimensional embeddings for visualization.
# 
# ##### Data Inspection: View input data like images or audio samples


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Extension for load tensorboard
get_ipython().run_line_magic('load_ext', 'tensorboard')


# import libraaries 
import tensorflow as tf
import datetime
import os



# check Current date & time
datetime.datetime.now()


# take the mist dataset
mnist = tf.keras.datasets.mnist


# prepare X,y
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print (X_train.shape)
print(X_test.shape)


# print image from dataset
import matplotlib.pyplot as plt
plt.imshow(X_train[66],cmap='gray_r')
plt.show()



# Normalise the dataset
X_train, X_test = X_train/255.0, X_test/255.0

# check values 
print ('X_train[0] :',X_train[0])
print('\n')
print ('X_test[0] :', X_test[0])



# define model in the function

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28),name="Flatten"),
        tf.keras.layers.Dense(512,activation='relu',name='Hidden_layer'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    

model = create_model()
model.summary()


# compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# .strftime(): This method formats the date and time into a string according to the specified fo



# Define a custom name
custom_name = "training_logs"

# Create a unique log directory path with custom name
log_dir = f"logs/fit/{custom_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"


# "logs/fit/": This is the base directory path where logs will be stored. 
# The fit part likely indicates that these logs are related to model fitting or training. 
# datetime.datetime.now().strftime("%Y%m%d-%H%M%S"): This part generates a timestamp based on the current date and time. 


# Create the TensorBoard callback

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)

# The TensorBoard callback is a tool to log various metrics and visualizations during model training, 
# log_dir is dynamically created 
# histogram_freq=1:means histograms will be logged at the end of every epoch.
# It helps monitor the training process, inspect the computational graph, and analyze model performance.


# train model

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),callbacks=[tensorboard_callback])
# The callbacks parameter is used to specify a list of callback functions that will be applied during training.
# launching Tensorboard inside Jupyter Notebook Environment

get_ipython().run_line_magic('tensorboard', '--logdir logs/fit --port=6006')


# The command %tensorboard --logdir logs/fit is used to launch TensorBoa rd directly within a Jupyter Notebook or Google Colab environment.
# This command points TensorBoard to the directory logs/fit, where the training logs (created by the TensorBoard callback during model training) are stored.
