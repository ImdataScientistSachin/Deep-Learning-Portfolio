#!/usr/bin/env python
# coding: utf-8

# ##  Part - I

# Load the Libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# Define the dataset URL
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" # tar ball

# Download and extract the dataset
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir

# Remove the suffix
data_dir = pathlib.Path(data_dir).with_suffix('')
print(data_dir)  # Output: path/to/file

# Import the glob module for pattern matching in file paths
import glob

# Count the number of JPEG images in the data directory and its subdirectories
image_count = len(list(data_dir.glob('*/*.jpg')))

# Print the total count of JPEG images found
print(image_count)

# Find all files and directories within the data_dir directory
data_dir.glob('*')


# Find all files and directories within the data_dir directory
list(data_dir.glob('*'))


# print roses directory

list(data_dir.glob('roses'))

# print all contains of roses directories 
# list(data_dir.glob('roses/*'))



# find length of rose directory 

rose_len  = len(list(data_dir.glob('roses/*')))

# print the contains with .jpg file format
rose_jpg = len(list(data_dir.glob('roses/*.jpg')))
tulips_len = len(list(data_dir.glob('tulips/*.jpg')))

print ('Rose Lenghth :',rose_len)
print ('Rose Jpg  :',rose_jpg)
print('Tulip Len', tulips_len)

# find the all jpg images

All_jpg = len(list(data_dir.glob('*/*.jpg')))
print ('All .jpg images :',All_jpg)

import os

# Specify the directory path

os.listdir('C:/Users/demog/.keras/datasets/flower_photos/')


# Explore the daise directories
os.listdir('C:/Users/demog/.keras/datasets/flower_photos/daisy/')


# find the length of the daise directory
daisy_len = len(os.listdir('C:/Users/demog/.keras/datasets/flower_photos/daisy/'))
print('size of daisy :' , daisy_len)


# open one of the image using pillow
PIL.Image.open('C:/Users/demog/.keras/datasets/flower_photos/daisy/10555815624_dc211569b0.jpg')


# load same Image using Load Image 
from tensorflow.keras.utils import load_img,img_to_array
load_img('C:/Users/demog/.keras/datasets/flower_photos/daisy/10555815624_dc211569b0.jpg')

# spacify the batch size , Height , Width
batch_size = 32
img_height = 180
img_width = 180


# Create a training dataset from the directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    # Use 20% of the data for validation
    validation_split=0.2,
    # Create a dataset for training
    subset="training",
    # Set a random seed for reproducibility
    seed=123,
    # Resize images to the specified dimensions
    image_size=(img_height, img_width),
    # Specify the batch size
    batch_size=batch_size
)


# Create a validation dataset from the directory

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    # Use 20% of the data for validation
    validation_split=0.2,
    # Create a dataset for validation
    subset="validation",
    # Set a random seed for reproducibility
    seed=123,
    # Resize images to the specified dimensions
    image_size=(img_height, img_width),
    # Specify the batch size
    batch_size=batch_size
)

# find class name
class_names = train_ds.class_names
print(class_names)



# Retrieve a sample batch from the training dataset
sample_img, labels = next(iter(train_ds))

# Print the shape of the sample images and labels
print("Sample Images Shape:", sample_img.shape)
print("Labels Shape:", labels.shape)


# Convert the tensor to a NumPy array with uint8 data type
sample_img.numpy().astype('uint8')
labels


# [print 0  index image ]
plt.imshow(sample_img[0].numpy().astype('uint8'))
print(labels[0])
class_names[0]


# plot the distribution to view multiple images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# prepare model

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5)
])
model.summary()


# Compile model 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# traain model

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# plot the distribution
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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


# download one of the image
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"



sunflower_path = tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url)
sunflower_path


PIL.Image.open('C:\\Users\\demog\\.keras\\datasets\\Red_sunflower.jpg')


# load the image
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

img
img_array = tf.keras.utils.img_to_array(img)
img_array
img_array.shape



# reshape the image 

img_array = img_array.reshape(1,180,180,3)
img_array.shape


# check the prediction
predictions = model.predict(img_array)
print (predictions[0])


# Apply softmax to the first set of predictions
score = tf.nn.softmax(predictions[0])
print( score.numpy().round(4))
print(class_names)
np.argmax(score.numpy())


# find the class 
class_names[np.argmax(score.numpy())]

# checkk prediction 

Pre_img = tf.keras.utils.load_img(
    'Rose1.jpeg', target_size=(img_height, img_width)
)
Pre_img


# covert to array
img_array = tf.keras.utils.img_to_array(Pre_img)
img_array = img_array.reshape(1,180, 180, 3)
img_array.shape


# check the predicton

predictions = model.predict(img_array)
print(predictions)



# apply softmax
score = tf.nn.softmax(predictions[0])
score = class_names[np.argmax(score.numpy())]
print (score)

# ### Dropout Layer # Augmentation
# Create a data augmentation pipeline
data_augmentation = keras.Sequential(
    [
        # Randomly flip the image horizontally
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        # Randomly rotate the image by up to 10 degrees
        layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# Create a figure with a specified size
plt.figure(figsize=(10, 10))

# Retrieve the first batch of images from the training dataset
for images, _ in train_ds.take(1):
    # Generate nine augmented versions of the first image
    for i in range(9):
        # Apply data augmentation to the images
        augmented_images = data_augmentation(images)
        
        # Create a subplot for each augmented image
        ax = plt.subplot(3, 3, i + 1)
        
        # Display the first augmented image in the subplot
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        
        # Hide the axis for each subplot
        plt.axis("off")

# Show the plot
plt.show()

# prepare model
model_new = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(5, name="outputs")
])

model.summary()

# compile model
model_new.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ### save the model
# ##### callback saves the model at certain intervals during training

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


# Create a list of callbacks
callbacks = [
    # Add the ModelCheckpoint callback
    checkpoint
]

# train model
epochs = 50
history = model_new.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,callbacks=callbacks
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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