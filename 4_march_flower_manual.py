#!/usr/bin/env python
# coding: utf-8

# ## Apply deep learning model on flower_manual dataset

# import the libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os


# URL for the flower photos dataset in tarball format

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" 

# Download the dataset from the specified URL and extract it
# The get_file function automatically handles downloading and caching the file

data_dir = tf.keras.utils.get_file('flower_photos.tgz', origin=dataset_url, extract=True)

# find directory

data_dir 


# Assume data_dir is the path to the downloaded and extracted file
# Convert data_dir to a Path object and remove its file extension.

data_dir = pathlib.Path(data_dir).with_suffix('')

data_dir 


# count the number of JPEG images within subdirectories of the data_dir directory. 

import glob
image_count = len(list(data_dir.glob('*/*.jpg')))
print('the length of Directory is : ',image_count)


# List all files and directories directly inside data_dir

item = list(data_dir.glob('*'))
item

# check one of the class from dir

list(data_dir.glob('roses'))


# print roses directory
# find the shape of roses
print ('Size od roses :' ,len(list(data_dir.glob('roses/*'))))
print('\n')
print ( list(data_dir.glob('roses/*')))


# find classes or directory using os library

os.listdir('C:\\Users\\demog\\.keras\\datasets\\flower_photos/')


# print one the directory

P_dir = os.listdir('C://Users//demog//.keras//datasets//flower_photos/dandelion/')

print(P_dir)

# print one of the sample Image using PIL (pillow) .

PIL.Image.open('C://Users//demog//.keras//datasets//flower_photos/dandelion/1241011700_261ae180ca.jpg')

# print one of the sample Image using

from tensorflow.keras.utils import load_img,img_to_array


load_img('C://Users//demog//.keras//datasets//flower_photos/dandelion/6918170172_3215766bf4_m.jpg')

# Assign 

batch_size = 32
img_height = 180
img_width = 180


# prepare training dataset

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Create a training dataset from the images in data_dir
# - validation_split=0.2: Use 20% of the data for validation (the rest is used for training)
# - subset="training": Select the training subset (as opposed to validation)
# - seed=123: Ensure reproducibility by setting a seed for shuffling
# - image_size=(img_height, img_width): Resize images to the specified dimensions
# - batch_size=batch_size: Load images in batches of the specified size


## Create a validation dataset from the images in data_dir
# - validation_split=0.2: Use 20% of the data for validation
# - subset="validation": Select the validation subset (as opposed to training)
# - seed=123: Ensure reproducibility by setting a seed for shuffling
# - image_size=(img_height, img_width): Resize images to the specified dimensions
# - batch_size=batch_size: Load images in batches of the specified size

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Inspect the structure of the training dataset
print(train_ds)
# find class name

class_names = train_ds.class_names
print(class_names)
print (len(class_names))

# Extract a sample image and its corresponding label from the training dataset
# - next(iter(train_ds)): Get the first batch of images and labels from the dataset
sample_img, labels = next(iter(train_ds))

# Convert the sample image tensor to a NumPy array and ensure its data type is uint8
sample_img.numpy().astype('uint8')

# Print the labels of the sample images
print (labels)

# print one the Image
plt.imshow(sample_img[27].numpy().astype('uint8'))

labels[27]

# print class name of that image
print(class_names[2])


# ploting SubPlot 

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# Define AUTOTUNE for dynamic buffer size in prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Cache and shuffle the training dataset, then prefetch with dynamic buffer size
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Cache and prefetch the validation dataset with dynamic buffer size
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# prepare model

model = Sequential([
# Rescale input images to have pixel values between 0 and 1
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
# Max pooling layer to reduce spatial dimensions
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
# Flatten the output of convolutional and pooling layers
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
# Output layer with 5 units (for a 5-class classification problem)
  layers.Dense(5)
])
model.summary()


# Compile Model without  using the 'Softmax' Activation

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# train model

epochs=15
history = model.fit( train_ds, validation_data=val_ds, epochs=epochs)


# Extract accuracy and loss metrics from training history
acc = history.history['accuracy']  # Training accuracy at each epoch
val_acc = history.history['val_accuracy']  # Validation accuracy at each epoch

loss = history.history['loss']  # Training loss at each epoch
val_loss = history.history['val_loss']  # Validation loss at each epoch

# Define the range of epochs for plotting
epochs_range = range(epochs)

# Create a figure with two subplots
plt.figure(figsize=(8, 4))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='center')  # Position legend at lower right corner
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='center')  # Position legend at upper right corner
plt.title('Training and Validation Loss')

# Display the plot
plt.show()

# Download a sample image of a sunflower for prediction

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"



Sunfpath = tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url)
Sunfpath

# open image using PIL
PIL.Image.open('C:\\Users\\demog\\.keras\\datasets\\Red_sunflower.jpg')

# Load and resize the image to the target dimensions
# modified image with custom dimention

img = tf.keras.utils.load_img(
    Sunfpath, target_size=(img_height, img_width)
)
# display image
img



# convert img  to numpy array

img_array = tf.keras.utils.img_to_array(img)

print (img_array)
print ('Img shape :' ,img_array.shape)
img_array = img_array.reshape(1,180,180,3)
print ('Img reshape :' ,img_array.shape)


# check Prediction

predictions = model.predict(img_array)
print('\n')

print('predictions with array ',predictions)
print('predictions outside the array ',predictions[0])



# Check Scores with Apply Softmax activation

score = tf.nn.softmax(predictions[0])

# conver prediction into numpy
n_score = score.numpy().round()
print('n_score',n_score)

# find higher values in array
print ('argmax val :',n_score.argmax())

# print class names
class_names



# print class name
class_names[np.argmax(score.numpy())]


# In[175]:


# load image from sample


img = tf.keras.utils.load_img(
    'Rose1.jpeg', target_size=(img_height, img_width)
)

img


# Conver img into array

img_array = tf.keras.utils.img_to_array(img)

print (img_array)
print ('Img shape :' ,img_array.shape)

# Reshape it
img_array = img_array.reshape(1,180, 180, 3)



# check prediction with this Image

predictions = model.predict(img_array)

print(predictions.round())
print('\n')
score = tf.nn.softmax(predictions[0])
score

# print class name
class_names[np.argmax(score.numpy())]

# save model