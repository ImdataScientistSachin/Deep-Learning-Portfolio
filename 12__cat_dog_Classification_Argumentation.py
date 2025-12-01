#!/usr/bin/env python
# coding: utf-8

#  Cat and Dog image classification using Deep Learning with Argumentation



# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)



# import the libraries 

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


# url path of the dataset
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'


# Download and UnZip the folder
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=URL, extract=True)


# find the path of directory & save into variable 
print (path_to_zip)

File_PATH = os.path.join(os.path.dirname(path_to_zip),'cats_and_dogs_filtered')
File_PATH

# list the directories inside the 'cats_and_dogs_filtered'
os.listdir('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered')

# we found train , validation , vectorize.py

# Or we can use this 
os.listdir(File_PATH)
# inside the cats_and_dogs_filtered directory we have train and validation directory

# dig inside the cats_and_dogs_filtered
os.listdir('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered\\train')


# 2 sub directory present Cat and dog

# explore train directory
os.listdir('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered\\train\\cats')

# we extract the training and validation data from File_PATH


# training directory
train_dir = os.path.join(File_PATH, 'train')

# list the directories from the train_dir
os.listdir(train_dir)

# validatio directory
validation_dir = os.path.join(File_PATH, 'validation')



# list the directories from the validation_dir
os.listdir(validation_dir)



# predefine Batch Size , Image size (beacause images are different shapes)
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
# we will resize all the images to 160x160

# import library to load images and convert them inti array
from tensorflow.keras.utils import load_img,img_to_array



# Load one of the image from cat directory
load_img('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered\\train\\cats\\cat.148.jpg')
load_img('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered\\train\\cats\\cat.111.jpg')



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
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

# Path to the directory containing the validation images
# Whether to shuffle the dataset before creating batches
# find the classes from  the dataset

class_names = train_dataset.class_names
print (class_names)

# visualize some images from the dataset
train_dataset.take(1)



# plot the images from the dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# check the shape of the image and labels from the dataset
for images, labels in train_dataset.take(1):
    print(images)
    print(labels)



for images, labels in train_dataset.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels[0])

# Convert the first image in the batch to a numpy array and cast it to uint8 format
# This is typically done to ensure the image data is in a suitable format for display or processing

# Check the number of validation batches
val_batches = tf.data.experimental.cardinality(validation_dataset)


#  Cardinality: Ensure that val_batches is known and not infinite or unknown. If it's unknown, 
# you might need to manually count the elements or ensure that the dataset is finite.

val_batches


# Split the validation dataset into a test dataset and a new validation dataset
# Take one-fifth of the validation dataset for testing
test_dataset = validation_dataset.take(val_batches // 5)

# Skip the portion taken for testing to create the new validation dataset
validation_dataset = validation_dataset.skip(val_batches // 5)


# Print the number of validation batches
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))

# Print the number of test batches
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

# Prefetching the datasets to improve performance during training and evaluation
# AUTOTUNE allows TensorFlow to dynamically adjust the prefetch buffer size based on system resources
"""
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
"""


# Define the AUTOTUNE constant for dynamic prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Prefetch the training dataset to improve performance
# AUTOTUNE dynamically adjusts the prefetch buffer size based on available system resources
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the validation dataset
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the test dataset
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ## Apply Argumentation
# Define a data augmentation pipeline using Keras Sequential API

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal_and_vertical',input_shape=(160,160,3)),
  tf.keras.layers.RandomRotation(0.2),
])  

# Randomly flip images both horizontally and vertically
# Loop through the first batch of the train dataset
for image, _ in train_dataset.take(1):
    # Create a new figure for plotting with a size of 10x10 inches
    plt.figure(figsize=(10, 10))
    
    # Extract the first image from the batch
    first_image = image[0]

    # Loop nine times to generate and display different augmented versions
    for i in range(9):
        # Create a subplot in a 3x3 grid
        ax = plt.subplot(3, 3, i + 1)
        
        # Apply data augmentation to the first image
        # tf.expand_dims adds a batch dimension, which is typically required for data augmentation operations
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        
        # Display the augmented image
        # Dividing by 255.0 normalizes pixel values to the range [0, 1], suitable for displaying images
        plt.imshow(augmented_image[0] / 255.0)
        
        # Hide axis ticks and labels for each subplot
        plt.axis('off')

# Show the plot (if not shown automatically)
plt.show()



# Extract the first batch from the dataset
image_batch, label_batch = next(iter(train_dataset))

# Print the shape of the image batch
print("Image batch shape:", image_batch.shape)


# show the labels
label_batch


# show the type of the image
type(image_batch[0])


# Display the fourth image from the batch
plt.imshow(image_batch[3].numpy().astype('uint8'))
plt.show()

# .astype('uint8'): This converts the data type of the array to unsigned 8-bit integers.
# .numpy(): This converts the TensorFlow tensor to a NumPy array.
# Matplotlib's imshow function can handle NumPy arrays directly.


# Import necessary Keras modules for building the model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling,Conv2D,MaxPooling2D,Flatten,Dense

# Define the model
model = Sequential([
    data_augmentation,
  Rescaling(1./255),
  Conv2D(16, 3, padding='same', activation='relu'), 
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1,activation='sigmoid')
])

# Define the neural network model using a sequential architecture.
# Apply data augmentation to the input images.
# Normalize pixel values from the range [0, 255] to [0, 1] .
# First convolutional layer with 16 filters, kernel size 3x3, and ReLU activation.
# First max pooling layer to reduce spatial dimensions . 
# Second convolutional layer with 32 filters, kernel size 3x3, and ReLU activation.
# Second max pooling layer to reduce spatial dimensions.
# Third convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
# Flatten the output of the convolutional and pooling layers into a 1D array.
# First dense layer with 128 neurons and ReLU activation . 
# Output layer with a single neuron and sigmoid activation for binary classification.





# check models summary
model.summary()


# Set the base learning rate
base_learning_rate = 0.0001

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])



# Set the number of initial epochs
initial_epochs = 100

# Train the model using the training dataset and validate on the validation dataset
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)



# Extract training and validation accuracy from the model's training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Extract training and validation loss from the model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a new figure with specified size for plotting
plt.figure(figsize=(4, 8))

# Plot training and validation accuracy in the first subplot
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='upper right')  # Display legend at the lower right corner
plt.ylabel('Accuracy')  # Set y-axis label
plt.ylim([min(plt.ylim()),1])  # Set y-axis limits to ensure accuracy values are visible
plt.title('Training and Validation Accuracy')  # Set title for the subplot

# Plot training and validation loss in the second subplot
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')  # Plot training loss
plt.plot(val_loss, label='Validation Loss')  # Plot validation loss
plt.legend(loc='lower right')  # Display legend at the upper right corner
plt.ylabel('Cross Entropy')  # Set y-axis label (commonly used for loss functions like cross-entropy)

# Dynamically set the y-axis limits for the loss plot
max_loss = max(loss + val_loss)  # Find the maximum loss value
margin = 0.1 * max_loss  # Add a 10% margin for better visualization
plt.ylim([0, max_loss + margin])  # Set y-axis limits

plt.title('Training and Validation Loss')  # Set title for the subplot
plt.xlabel('epoch')  # Set x-axis label

# Display the plot
plt.show()



# load Image for prediction
img = load_img('predict_dog_3.jpeg',target_size=(160,160))
img_C = load_img('predict_cat.jpeg',target_size=(160,160))
(img)


(img_C)


# convert tensor image to array
img_arr = img_to_array(img)
img_arr_C = img_to_array(img_C)

print (img_arr)
print (img_arr_C)



# check shape
img_arr.shape


# Reshape the array to the desired format
img_arr = img_arr.reshape(1,160,160,3)
img_arr_C = img_arr_C.reshape(1,160,160,3)



# Check Prediction (dog)
model.predict(img_arr)


# Check Prediction (cat)
model.predict(img_arr_C)
# if the output is near to 1  it is dog
# if the output is near to 0 it is cat