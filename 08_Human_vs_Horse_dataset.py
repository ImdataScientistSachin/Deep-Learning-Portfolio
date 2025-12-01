#!/usr/bin/env python
# coding: utf-8

# ## classification Between the Humans and Horse

# import the libraries

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds  # we use tenserflow dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model

import warnings
warnings.filterwarnings('ignore')



data, info = tfds.load("horses_or_humans", split=['train', 'test'], shuffle_files=False, as_supervised=True, with_info=True)

# Load the 'horses_or_humans' dataset, splitting it into 'train' and 'test' sets.
# The 'shuffle_files=False' parameter prevents shuffling of the files during loading.
# 'as_supervised=True' ensures that the data is returned as input-label pairs.
# 'with_info=True' provides additional information about the dataset.



print(data)
print(info)



# Extract the class names from the dataset's metadata
classes = info.features['label'].names

# Get the number of classes in the dataset
n_classes = info.features['label'].num_classes

# Print the class names and the number of classes for inspection
print("Class Names:", classes)
print("Number of Classes:", n_classes)


# Assign the training dataset from the loaded data
train_ds = data[0]

# Assign the test dataset from the loaded data
test_ds = data[1]
train_ds
test_ds


# Define a function to display images from a dataset
def display_dataset_images(ds, nrows=3, ncols=4, figsize=(16,12)):
    """
    Display images from a dataset in a grid layout.

    Parameters:
    - ds: Dataset to display images from.
    - nrows: Number of rows in the grid.
    - ncols: Number of columns in the grid.
    - figsize: Figure size in inches.
    """
    
    # Create a figure with subplots based on nrows and ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Flatten the axs array for easier indexing
    axs = np.ravel(axs)
    
    # Initialize a counter to keep track of the current subplot index
    i = 0
    
    # Iterate over the dataset, taking as many images as there are subplots
    for image, label in ds.take(len(axs)):
        # Set the current subplot as the active one
        plt.sca(axs[i])
        
        # Display the image
        plt.imshow(image.numpy())
        
        # Set the title with the class name and image dimensions
        plt.title(f"{classes[label]} - {image.numpy().shape}")
        
        # Turn off axis ticks for cleaner display
        plt.axis('off')
        
        # Increment the subplot index
        i += 1
    
    # Show the plot
    plt.show()



# displaay train data
display_dataset_images(train_ds)

# display test data
display_dataset_images(test_ds)


# Define a constant for automatic tuning of prefetch buffer size
AUTOTUNE = tf.data.AUTOTUNE

# Optimize the training dataset pipeline
train_ds = (
    # Shuffle the dataset to randomize the order of examples
    train_ds.shuffle(256)
    # Batch the dataset into groups of 32 examples
    .batch(32)
    # Prefetch batches to improve performance by overlapping computation and data loading
    .prefetch(buffer_size=AUTOTUNE)
)

# Optimize the test dataset pipeline
test_ds = (
    # Batch the dataset into groups of 32 examples
    test_ds.batch(32)
    # Prefetch batches to improve performance
    .prefetch(buffer_size=AUTOTUNE)
)

# prepare Model 

model = Sequential([
  Rescaling(1./255,input_shape=(300,300,3)),
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

model.summary()

# Compile the model with the specified optimizer, loss function, and metrics
model.compile(
   
    optimizer='adam',
    
    # Use binary cross-entropy loss, suitable for binary classification tasks
    loss='binary_crossentropy',
    
    # Track accuracy as a metric during training and evaluation
    metrics=['accuracy']
)


# train the model 

history = model.fit(train_ds, epochs=50, 
                    validation_data=test_ds)


# In[31]:


# Extract accuracy and loss histories from the training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Define the range of epochs for plotting
epochs_range = range(len(acc))  # Ensure this matches the actual number of epochs

# Create a figure with two subplots
plt.figure(figsize=(8, 4))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc=0)  # Display legend at the best location
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc=0)  # Display legend at the best location
plt.title('Training and Validation Loss')

# Show the plot
plt.tight_layout()  # Ensure titles fit within the figure area
plt.show()


# load the train data images
img = tf.keras.utils.load_img(
    'Horse2.jpeg', target_size=(300, 300))
img


# convert the tensor image to array

img_array = tf.keras.utils.img_to_array(img)
img_array.shape


# Create a batch by adding a new dimension (batch dimension) to the image array
img_array = tf.expand_dims(img_array, 0)

# Print the shape of the resulting array to verify the batch dimension
print(img_array.shape)


# check prediction
predictions = model.predict(img_array)
print(predictions[0].round())