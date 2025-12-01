#!/usr/bin/env python
# coding: utf-8

""" # # Fashin_MNIST dataset """

# ##### The Fashion-MNIST dataset is a collection of 70,000 grayscale images of fashion products from Zalando, sized at 28x28 pixels.Same as MNIST datset.

# ##### Labels: 0: T-shirt/top ,1: Trouser, 2: Pullover, 3: Dress , 4: Coat
# #####         5: Sandal , 6: Shirt , 7: Sneaker, 8: Bag, 9: Ankle boot

# ## Practicle Implementation 

# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import Libraries

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # Include load_model if needed
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Print versions to verify installations
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Import deployment packages (if needed)
import joblib  # Note: Use joblib for non-Keras models only.

# Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# List of class names - this MUST match your categories in index.html
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize pixel values

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0


# Reshape to include channel dimension
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_images.shape
test_images.shape

# show dataset details
train_images[0]
train_images[0].shape

# same as MNIST dataset
train_labels



# plot the distribution
plt.imshow(train_images[0],cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()

# show label
train_labels[0]


# plot the distributions

plt.imshow(train_images[0],cmap='gray_r')
plt.title(train_labels[0])
plt.xticks([])
plt.yticks([])
plt.show()



plt.imshow(train_images[70],cmap='gray_r')
plt.title(train_labels[70])
plt.xticks([])
plt.yticks([])
plt.show()


# create a Variable to store labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names[6]
class_names[train_labels[0]]

# plot the distribution with actual class name

plt.imshow(train_images[70],cmap='gray_r')
plt.title(class_names[train_labels[70]])
plt.xticks([])
plt.yticks([])
plt.show()


# create distribution with multiple subplots

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
# plt.cm.binary is a colormap that displays the image in shades of black and white (binary).
    plt.xlabel(class_names[train_labels[i]])
plt.tight_layout()
plt.show()

# show first image details
train_images[0]

"""

# Model Creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
"""
# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()


# compile the model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Early Stopping Callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

# 'val_loss': Monitors the validation loss.
# 'loss': Monitors the training loss.
# patience: This parameter indicates how many epochs with no improvement should be allowed before stopping the training.
# min_delta: This defines the minimum change in the monitored quantity to qualify as an improvement. 
# If the change is less than this threshold, it will not be considered an improvement.

# verbose: Controls the verbosity of the output. Setting it to 1 will display messages when stopping occurs.

# mode: This can be set to 'min', 'max', or 'auto'. 
# For metrics like loss, you would typically use 'min', indicating that training should stop when the monitored quantity has stopped decreasing. 
# restore_best_weights: If set to True, this option restores model weights from the epoch with the best monitored metric after training has stopped.  



# trian the model

# history = model.fit(train_images, train_labels,validation_split=0.2, epochs=20)


history = model.fit(train_images, train_labels, epochs=20, batch_size=32,
                    validation_data=(test_images, test_labels))


history.history
history.history['loss']
history.history['accuracy']
history.history['val_loss']
history.history['val_accuracy']


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
# Assume `model` is your trained model variable
# in our case is "history ".


from tensorflow.keras.models import Sequential  # or your specific model type
# Assuming 'model' is your trained Keras model
model.save('Appl_history.h5')  # Save as HDF5 format



# Step 1: Verify the File Location

import os

# Check if the file exists
filename = 'Appl_history.h5'
current_directory = os.getcwd()
file_path = os.path.join(current_directory, filename)

if os.path.exists(file_path):
    print(f"File found at: {file_path}")
else:
    print(f"File not found at: {file_path}")



# Step 2: Load the Model
"""
# in above the epochs are runs = 13


# Extract data from history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Determine the number of epochs based on the actual training history
epochs = len(acc)
epochs_range = range(epochs)

# Print lengths for debugging (optional but helpful)
print(f"Number of Epochs: {epochs}")
print(f"Length of Training Accuracy: {len(acc)}")
print(f"Length of Validation Accuracy: {len(val_acc)}")
print(f"Length of Training Loss: {len(loss)}")
print(f"Length of Validation Loss: {len(val_loss)}")

# Plot the training and validation curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')  # More specific location
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Accuracy')  # Add y-axis label

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')  # More specific location
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Loss')  # Add y-axis label

plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.show()

"""

# Save the model
model.save('Appl_history.h5')

# Plot training history (optional)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Load the model
# test prediction


#plt.figure(figsize=(10,10))
plt.imshow(test_images[1250],cmap='gray_r')
plt.title(class_names[test_labels[1250]])
plt.xticks([])
plt.yticks([])
plt.show()


# Test prediction
model.predict(test_images[1250].reshape(1,28,28))

# reshape(1,28,28)) = 1-image , 28*28 size 
model.predict(test_images[1250].reshape(1,28,28)).round(3)


# check Prediction
np.argmax(model.predict(test_images[1250].reshape(1,28,28)))


# check Prediction 2
class_names[np.argmax(model.predict(test_images[1250].reshape(1,28,28)))]