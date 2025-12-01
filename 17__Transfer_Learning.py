#!/usr/bin/env python
# coding: utf-8

# ### Horse vs Human dataset Transfer Learning 

# #####  Transfer learning involves taking a model that has been pre-trained on a large dataset and fine-tuning it for a specific task that may have limited data. The core idea is to utilize the features and representations learned during the initial training to enhance the model's ability to generalize to new, but related tasks

# ### Methods of Transfer Learning
# 
# ##### Fine-Tuning Pre-Trained Models: This method involves taking a model trained on a large dataset (e.g., ImageNet) and adjusting its weights for a new, similar task. Typically, the earlier layers of the model, which capture general features, are frozen, while the later layers are retrained .
# 
# ##### Feature Extraction: In this approach, the pre-trained model is used as a feature extractor. The output from the earlier layers is fed into a new classifier tailored for the specific task. This method allows for quick adaptation without extensive retraining.
# 
# ##### Homogeneous Transfer Learning: This type deals with tasks that share similar feature spaces. For instance, if one domain has ample labeled data while another has limited data, knowledge can be transferred by aligning distributions between these domains.
# 
# ##### Domain Adaptation: This scenario occurs when the source and target domains differ significantly. Techniques are employed to adapt the source domain's characteristics to better fit the target domain's distribu.tiontion
# ##### Multi-Task Learning: In this method, a model is trained on multiple related tasks simultaneously. The shared representations learned from these tasks can improve performance on each individual task.

# import libraries

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model

import warnings
warnings.filterwarnings('ignore')



# split dataset
data, info = tfds.load("horses_or_humans", split=['train', 'test'], shuffle_files=False, as_supervised=True, with_info=True)

# 'data' will contain the actual dataset (train and test splits)
# 'info' will contain information about the dataset, such as number of classes, number of examples, etc.
# 'as_supervised=True' indicates that the data will be returned as (input, label) pairs.
# 'with_info=True' returns additional metadata about the dataset along with the data.

data
info


# print classes

classes = info.features['label'].names
n_classes = info.features['label'].num_classes
print(classes)
print(n_classes)



# define train, test
train_ds = data[0]
test_ds = data[1]
train_ds
test_ds

# function to display images from dataset
def display_dataset_images(ds, nrows=3, ncols=4, figsize=(16,12)):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = np.ravel(axs)
    i = 0
    for image, label in ds.take(len(axs)):
        plt.sca(axs[i])
        plt.imshow(image.numpy())
        plt.title(f"{classes[label]} - {image.numpy().shape}")
        plt.axis('off')
        i+=1
    plt.show()


# Create a figure and a set of subplots with specified number of rows and columns.
# Flatten the array of axes to easily iterate through them(row wise)
# Initialize a counter for images.
# Iterate through the dataset and take only as many images as there are axes.
# Set the current axis to display the image.
# Convert the image tensor to a numpy array and display it
# Set the title with the class name and shape of the image.
# Turn off axis labels for a cleaner look



# print the images
display_dataset_images(test_ds)


# function to process image and label
def image_process(image, label):
    image = keras.applications.mobilenet_v3.preprocess_input(image)
    label = tf.cast(label, tf.float32)
    return image, label

# Processes the input image and label for model training.
# Preprocess the image using MobileNetV3 preprocessing function
# This typically includes scaling pixel values to a range suitable for the model.
# Cast the label to float32 type for compatibility with model training


# Map the image_process function to the training, testing dataset

train_ds = train_ds.map(image_process).shuffle(256).batch(32).prefetch(1)

test_ds = test_ds.map(image_process).batch(32).prefetch(1)

# This applies the same image processing function to each (image, label) pair in the test dataset.
# Prefetch 1 batch to improve performance during training, testing

# Create the base model using MobileNetV3Large architecture
base_model = keras.applications.MobileNetV3Large(input_shape=(300, 300, 3), 
                                                 include_top=False, 
                                                 pooling='avg')

# Create the base model using MobileNetV3Large architecture.
# Specify the input shape of the images (height, width, channels)
# Exclude the fully connected layer at the top of the model.
# Use average pooling to reduce the feature maps to a single vector per image.


# find summary
base_model.summary()

# Freeze the base model's layers to prevent them from being updated during training
base_model.trainable = False

# Set the base model's trainable attribute to False
# This means that the weights of the base model will not be updated during training.
# This is typically done when using transfer learning to freeze the layers of a pre-trained model.

# Add custom layers on top of the base model for binary classification

# Add a fully connected (Dense) layer with 1024 units and ReLU activation function
x = Dense(1024, 'relu')(base_model.output)

# Apply dropout regularization with a rate of 0.5 to prevent overfitting
x = Dropout(0.5)(x)

# Add another fully connected layer with 256 units and ReLU activation function
x = Dense(256, 'relu')(x)

# Apply dropout regularization again, this time with a rate of 0.2
x = Dropout(0.2)(x)

# Add a fully connected layer with 64 units and ReLU activation function
x = Dense(64, 'relu')(x)

# Final output layer with 1 unit and sigmoid activation function for binary classification
out = Dense(1, 'sigmoid')(x)



# Load the base model (VGG16) without the top layers
model = Model(inputs=base_model.input, outputs=out)

# Create a new model that includes the base model and the custom layers added on top.
# Specify the inputs as the input of the base model and the outputs as the final output layer
model.summary()

# compile model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# use early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                               patience=5, 
                                               restore_best_weights=True)


# Metric to monitor during training (validation accuracy)
# patience=5, Number of epochs with no improvement after which training will be stopped .
# restore_best_weights=True # Restore model weights from the epoch with the best monitored value


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", 
                                              patience=5)

 # Metric to monitor (validation accuracy).
 # Number of epochs with no improvement before reducing the learning rate


# train model
history = model.fit(train_ds, epochs=20, 
                    validation_data=test_ds, 
                    callbacks=[early_stopping, reduce_lr], 
                    verbose=1)


# In[29]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(9)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# Plot training and validation accuracy and loss over epochs.
# Create a figure with two subplots: one for accuracy and one for loss.