#!/usr/bin/env python
# coding: utf-8

""" #  Intel Image Classification  """

# ##### This is image data of Natural Scenes around the world.
 
# ##### Contents:

# ##### This Data contains around 25k images of size 150x150 distributed under 6 categories.
# #####       {'buildings' -> 0,
#  #####       'forest' -> 1,
#  #####       'glacier' -> 2,
# #####        'mountain' -> 3,
# #####        'sea' -> 4,
# #####        'street' -> 5
# #####        }


# only for cuda enabled laptop and desktop
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Physical Devices:", physical_devices)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# # Load the Libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pathlib
import zipfile
import glob


Dir_Path = r"C:\\Users\\demog\\.keras\\datasets\\Intel_Image_Classification\\"
train_zip_path = os.path.join(Dir_Path, "seg_train.zip")
test_zip_path = os.path.join(Dir_Path, "seg_test.zip")

# train_dir = r"C:\\Users\\demog\\.keras\\datasets\\Intel_Image_Classification\\seg_train.zip"
# test_dir = r"C:\\Users\\demog\\.keras\\datasets\\Intel_Image_Classification\\seg_test.zip"


# Step 2: Check if zip files exist

if not os.path.exists(train_zip_path):
    print(f"Training zip file does not exist at {train_zip_path}")
if not os.path.exists(test_zip_path):
    print(f"Testing zip file does not exist at {test_zip_path}")
    
# Step 3: Print paths to verify
print(Dir_Path)
print(train_zip_path)
print(test_zip_path)

# list the directories inside the germantraffic signs 
directory = os.listdir(Dir_Path)
directory



train_dir = tf.keras.utils.get_file(
    fname="image_train.zip",
    origin=f"file:\\{train_zip_path}",
    extract=True,
    archive_format="zip"
)


test_dir = tf.keras.utils.get_file(
    fname="image_test.zip",
    origin=f"file:\\{test_zip_path}",
    extract=True,
    archive_format="zip"
)

# print the extracted paths

train_dir = os.path.join(Dir_Path, "seg_train")
test_dir = os.path.join(Dir_Path, "seg_test")

print(train_dir)
print(test_dir)


# explore train directory
train_contains = os.listdir(train_dir)
print(len(train_contains))
train_contains


# explore test directory
test_contains = os.listdir(test_dir)

# print one of the sample Image using
from tensorflow.keras.utils import load_img,img_to_array

forest = 'C:/Users/demog/.keras/datasets/Intel_Image_Classification/seg_train/forest/'
print('total images in forest :' ,len(forest))
print ('\n')
print("Files in directory:", os.listdir(forest))

# sample image path
image_path = r'C:\Users\demog\.keras\datasets\Intel_Image_Classification\seg_train\forest\12968.jpg'

# Load and display the image
if os.path.isfile(image_path):
    img = load_img(image_path)
    print("Image loaded successfully.")
else:
    print(f"File not found: {image_path}")
img


# Convert the image to a NumPy array and display its shape
img_array = img_to_array(img)
print("Image shape:", img_array.shape)

#  predefine Batch Size , Image size (beacause images are different shapes)
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

# Load the training dataset from the specified directory
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Path to the directory containing the training images
# Whether to shuffle the dataset before creating batches
# Number of images to include in each batch
# Resize all images to this size (width, height)

# Load the validation dataset from the specified directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)



# find the classes from  the dataset
class_names =train_dataset.class_names
class_names


# Retrieve a sample batch from the training dataset
sample_img, labels = next(iter(train_dataset))

# Print the shape of the sample images and labels
print("Sample Images Shape:", sample_img.shape)
print("Labels Shape:", labels.shape)

# Print the labels
labels


# [Print image as per index ]
index = 4  # We can change this to any valid index within the batch size
plt.imshow(sample_img[index].numpy().astype('uint8'))
plt.axis('off')  # Hide axes for better visualization
plt.show()  # Show the image

# Print the label and class name for the image at index 

print("Label:", labels[index].numpy())  # Convert tensor to NumPy for display
print("Class Name:", class_names[labels[index].numpy()])  # Use label as index to get class name



# plot the distribution to view multiple images

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i].numpy()])
    plt.axis("off")
   


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# prepare model
"""
model = Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),
    # Data augmentation (optional, but recommended)
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])
"""

model = Sequential([
    Input(shape=(150, 150, 3)),             # 1. Explicit Input layer
    Rescaling(1./255),
    Conv2D(32, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
    # Data augmentation to increase dataset diversity and prevent overfitting.
    # Rescale pixel values to be between 0 and 1 .
    # Convolutional layer with 32 filters, kernel size 5x5, and ReLU activation .
    # Batch normalization to stabilize learning .
    # Max pooling to reduce spatial dimensions and extract dominant features .
    # Convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
    # Batch normalization .
    # Max pooling.
    # Convolutional layer with 128 filters, kernel size 3x3, and ReLU activation .
    # Flatten the output of convolutional and pooling layers into a 1D vector .
    # Dropout to prevent overfitting by randomly dropping neurons
    # Dense layer with 6 units (one for each class) and softmax activation for multi-class classification .
    
# Print the model summary
model.summary()


# Compile model 
model.compile(optimizer= Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Callback for robust training
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# traain model
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
    callbacks=[early_stop]
)



# plot the distribution

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='center right')
plt.title('Training and Validation Loss')
plt.show()



# Pass dummy data to build the model

dummy_input = tf.random.normal([1, 150, 150, 3])
_ = model(dummy_input) 
# Forward pass to build the model


# -------- Save Model for Deployment --------

model.save("intel_image_Classifier_V1_model.h5")
print("Model saved as Intel_image_Classifier_V1_models.h5")