#!/usr/bin/env python
# coding: utf-8


# https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip


""" # ### working  on Googles germantrafficsigns dataset  """


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
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
import pathlib


# Define the dataset URL
dataset_url = " https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip"    



# Download and extract the dataset
data_dire = tf.keras.utils.get_file('germantrafficsigns.zip', origin=dataset_url, extract=True)

# Print the directory where the zip was extracted
print (data_dire)


# List the contents of the extracted directory to understand the structure
extracted_files = os.listdir(os.path.dirname(data_dire))
print("Files in extracted directory:", extracted_files)


File_PATH = os.path.join(os.path.dirname(data_dire),'germantrafficsigns')
File_PATH

# list the directories inside the germantrafficsigns 

# os.listdir('C:\\Users\\demog\\.keras\\datasets\\')
directory = os.listdir(File_PATH)
directory

# Find all files and directories within the data_dir directory
# dig inside the germantrafficsigns directory

# train_dir = 'C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\train'

train_dir = os.path.join(File_PATH, 'train')
train_dir


# validation_dir = 'C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\validation'

validation_dir = os.path.join(File_PATH, 'validation')
validation_dir

# explore train directory
train_contains = os.listdir(train_dir)
train_contains

print(len(train_contains))
train_contains

# explore validation directory
validation_contains = os.listdir(validation_dir)

print(len(validation_contains))
validation_contains

# predefine Batch Size , Image size (beacause images are different shapes)
BATCH_SIZE = 32
IMG_SIZE = (180, 180)


# import library to load images and convert them inti array

from tensorflow.keras.utils import load_img,img_to_array


# Load one of the image
load_img('C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\train\\00008\\00000_00023.jpg')

# Load the training dataset from the specified directory

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Path to the directory containing the training images
# Whether to shuffle the dataset before creating batches
# Number of images to include in each batch
# Resize all images to this size (width, height)


validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
# Load the validation dataset from the specified directory
# find the classes from  the dataset

class_names =train_dataset.class_names
print (class_names)

# find the length of classes
No_classes=len(class_names)
print(No_classes)

# Apply one-hot encoding directly using map() and ensure labels are cast to int32

train_dataset = train_dataset.map(lambda images, labels: (images, tf.one_hot(tf.cast(labels, tf.int32), depth=No_classes)))
validation_dataset = validation_dataset.map(lambda images, labels: (images, tf.one_hot(tf.cast(labels, tf.int32), depth=No_classes)))

train_dataset.take(1)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    # Convert one-hot encoded labels to scalar indices
    class_indices = tf.argmax(labels, axis=1).numpy()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[class_indices[i]])  # Use class_indices for indexing
        plt.axis("off")


for images, labels in train_dataset.take(1):
    print(images)
    print(labels)


for images, labels in train_dataset.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels[0])

# Convert the first image in the batch to a numpy array and cast it to uint8 format
# This is typically done to ensure the image data is in a suitable format for display or processing


for images, labels in train_dataset.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels[0])
# Convert the first image in the batch to a numpy array and cast it to uint8 format
# This is typically done to ensure the image data is in a suitable format for display or processing

# Create test dataset from validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset).numpy()


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


# Define the AUTOTUNE constant for dynamic prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Prefetch the training dataset to improve performance
# AUTOTUNE dynamically adjusts the prefetch buffer size based on available system resources
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the validation dataset
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the test dataset
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling,Conv2D,MaxPooling2D,Flatten,Dense

# Define the model
model = Sequential([
    layers.Input(shape=(180, 180, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(43, activation='softmax') 
])

# Define the neural network model using a sequential architecture.
# Normalize pixel values from the range [0, 255] to [0, 1] .
# First convolutional layer with 32 filters, kernel size 3x3, and ReLU activation.
# First max pooling layer to reduce spatial dimensions . 
# Second convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
# Second max pooling layer to reduce spatial dimensions.
# Flatten the output of the convolutional and pooling layers into a 1D array.
# First dense layer with 256 neurons and ReLU activation . 
# Output layer with a 43 neuron and softmax activation for binary classification.



model.summary()

# Set the base learning rate
base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])



# Train the model

history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset)


# Extract training and validation accuracy from the model's training history
acc = history.history['categorical_accuracy']  # Change 'accuracy' to 'categorical_accuracy'
val_acc = history.history['val_categorical_accuracy']  # Change 'val_accuracy' to 'val_categorical_accuracy'

# Extract training and validation loss from the model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a new figure with specified size for plotting
plt.figure(figsize=(4, 8))

# Plot training and validation accuracy in the first subplot
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='upper right')  # Display legend at the upper right corner
plt.ylabel('Accuracy')  # Set y-axis label
plt.ylim([min(plt.ylim()), 1])  # Set y-axis limits to ensure accuracy values are visible
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
plt.xlabel('Epoch')  # Set x-axis label

# Display the plot
plt.show()



# load Image for prediction

img = load_img('diging pit.jpeg',target_size=(180,180))
img_C = load_img("50_speed_limit.jpeg",target_size=(180,180))

(img)
img_C


# convert tensor image to array
img_arr = img_to_array(img)
img_arr_C = img_to_array(img_C)


# check image shape

print(img_arr.shape)
print(img_arr_C.shape)

# print image array
print (img_arr)
print (img_arr_C)


"""
# Reshape the array to the desired format
img_arr = img_arr.reshape(1,180,180,3)
img_arr_C = img_arr_C.reshape(1,180,180,3)

"""


# or Use Below Command



# Expand dimensions to include batch size (1, height, width, channels)
img_arr = np.expand_dims(img_arr, axis=0)

img_arr_C = np.expand_dims(img_arr_C, axis=0)

# Add batch dimension



# check image shape after expand dimention

print(img_arr.shape)
print(img_arr_C.shape)



# Normalize the image arrays (if required by your model)
img_arr /= 255.0
img_arr_C /= 255.0



# Normalize pixel values



# Debugging: Print shapes and sample pixel values
print("Shape of img_arr:", img_arr.shape)
print("Shape of img_arr_C:", img_arr_C.shape)
print("Sample pixel value from img_arr:", img_arr[0][0][0])



# Make predictions (remove .round() temporarily for debugging)
m_pred1 = model.predict(img_arr)  # Prediction for the first image
m_pred2 = model.predict(img_arr_C)  # Prediction for the second image


# Get the class indices with the highest probability
pred_class1 = m_pred1.argmax()  # Class index for the first image

pred_class2 = m_pred2.argmax()  # Class index for the second image
print(f"Predicted class for 'diging pit.jpeg': {pred_class1}")

print(f"Predicted class for 'speed limit .jpg': {pred_class2}")



"""
# full code # 

#  Load the Libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# Define the dataset URL

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip"    

# Download and extract the dataset

data_dire = tf.keras.utils.get_file('germantrafficsigns.zip', origin=dataset_url, extract=True)

# Print the directory where the zip was extracted

print (data_dire)

# List the contents of the extracted directory to understand the structure
extracted_files = os.listdir(os.path.dirname(data_dire))
print("Files in extracted directory:", extracted_files)

File_PATH = os.path.join(os.path.dirname(data_dire),'germantrafficsigns')

File_PATH

# list the directories inside the germantrafficsigns 

# os.listdir('C:\\Users\\demog\\.keras\\datasets\\')
directory = os.listdir(File_PATH)

directory

# dig inside the germantrafficsigns directory

# train_dir = 'C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\train'

train_dir = os.path.join(File_PATH, 'train')

# validation_dir = 'C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\validation'

validation_dir = os.path.join(File_PATH, 'validation')


# explore train directory
train_contains = os.listdir(train_dir)


train_contains

print(len(train_contains))
train_contains


# predefine Batch Size , Image size (beacause images are different shapes)

BATCH_SIZE = 32
IMG_SIZE = (180, 180)
EPOCHS = 50

# import library to load images and convert them inti array

from tensorflow.keras.utils import load_img,img_to_array


# Load one of the image

load_img('C:\\Users\\demog\\.keras\\datasets\\germantrafficsigns\\train\\00008\\00000_00023.jpg')

# Load the training dataset from the specified directory

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Path to the directory containing the training images
# Whether to shuffle the dataset before creating batches
# Number of images to include in each batch
# Resize all images to this size (width, height)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)



# find the classes from  the dataset

class_names = train_dataset.class_names
print (class_names)

# find the length of classes
No_classes=len(class_names)
print(No_classes)


# Apply one-hot encoding directly using map() and ensure labels are cast to int32

train_dataset = train_dataset.map(lambda images, labels: (images, tf.one_hot(tf.cast(labels, tf.int32), depth=No_classes)))
validation_dataset = validation_dataset.map(lambda images, labels: (images, tf.one_hot(tf.cast(labels, tf.int32), depth=No_classes)))
train_dataset.take(1)


plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


for images, labels in train_dataset.take(1):
    print(images)
    print(labels)


for images, labels in train_dataset.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels[0])

# Convert the first image in the batch to a numpy array and cast it to uint8 format
# This is typically done to ensure the image data is in a suitable format for display or processing

for images, labels in train_dataset.take(1):
    print(images[0].numpy().astype('uint8'))
    print(labels[0])

# Convert the first image in the batch to a numpy array and cast it to uint8 format
# This is typically done to ensure the image data is in a suitable format for display or processing

# Create test dataset from validation dataset

val_batches = tf.data.experimental.cardinality(validation_dataset).numpy()
val_batches 



#  Cardinality: Ensure that val_batches is known and not infinite or unknown. If it's unknown, 
# you might need to manually count the elements or ensure that the dataset is finite.


# Split the validation dataset into a test dataset and a new validation dataset
# Take one-fifth of the validation dataset for testing

test_dataset = validation_dataset.take(val_batches // 5)

# Skip the portion taken for testing to create the new validation dataset
validation_dataset = validation_dataset.skip(val_batches // 5)



# Print the number of validation batches
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))

# Print the number of test batches
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))



# Define the AUTOTUNE constant for dynamic prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Prefetch the training dataset to improve performance
# AUTOTUNE dynamically adjusts the prefetch buffer size based on available system resources
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the validation dataset
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Prefetch the test dataset
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling,Conv2D,MaxPooling2D,Flatten,Dense

# Define the model
model = Sequential([
    layers.Input(shape=(180, 180, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(43, activation='softmax') 
])

# Define the neural network model using a sequential architecture.
# Normalize pixel values from the range [0, 255] to [0, 1] .
# First convolutional layer with 16 filters, kernel size 3x3, and ReLU activation.
# First max pooling layer to reduce spatial dimensions . 
# Second convolutional layer with 32 filters, kernel size 3x3, and ReLU activation.
# Second max pooling layer to reduce spatial dimensions.
# Third convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
# Flatten the output of the convolutional and pooling layers into a 1D array.
# First dense layer with 128 neurons and ReLU activation . 
# Output layer with a 43 neuron and softmax activation for categorical classification.






# Set the base learning rate
base_learning_rate = 0.0001

# Compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])


# Train the model

history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset)




# Extract training and validation accuracy from the model's training history
acc = history.history['categorical_accuracy']  # Change 'accuracy' to 'categorical_accuracy'
val_acc = history.history['val_categorical_accuracy']  # Change 'val_accuracy' to 'val_categorical_accuracy'

# Extract training and validation loss from the model's training history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a new figure with specified size for plotting
plt.figure(figsize=(4, 8))

# Plot training and validation accuracy in the first subplot
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='upper right')  # Display legend at the upper right corner
plt.ylabel('Accuracy')  # Set y-axis label
plt.ylim([min(plt.ylim()), 1])  # Set y-axis limits to ensure accuracy values are visible
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
plt.xlabel('Epoch')  # Set x-axis label

# Display the plot
plt.show()



# load Image for prediction

img = load_img('diging pit.jpeg',target_size=(180,180))
img_C = load_img("No left turn.jpeg",target_size=(180,180))
(img)


# convert tensor image to array
img_arr = img_to_array(img)
img_arr_C = img_to_array(img_C)


# Expand dimensions to include batch size (1, height, width, channels)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr_C = np.expand_dims(img_arr_C, axis=0)

# check image shape after expand dimention

print(img_arr.shape)
print(img_arr_C.shape)


# Normalize the image arrays (if required by your model)
img_arr /= 255.0
img_arr_C /= 255.0

# Debugging: Print shapes and sample pixel values
print("Shape of img_arr:", img_arr.shape)
print("Shape of img_arr_C:", img_arr_C.shape)
print("Sample pixel value from img_arr:", img_arr[0][0][0])

# check prediction

m_pred1 = model.predict(img_arr)  # Prediction for the first image
m_pred2 = model.predict(img_arr_C)  # Prediction for the second image

# Get the class indices with the highest probability
pred_class1 = m_pred1.argmax()  # Class index for the first image

pred_class2 = m_pred2.argmax()  # Class index for the second image



print(f"Predicted class for 'diging pit.jpeg': {pred_class1}")


print(f"Predicted class for 'speed limit .jpg': {pred_class2}")


"""

