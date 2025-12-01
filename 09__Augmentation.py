#!/usr/bin/env python
# coding: utf-8

#  Augmentation in Deep Learning

# ### A technique to increase the diversity of your training set by applying random (but realistic) transformations, such as image rotation.
# 
# ##### Data augmentation is a crucial technique in deep learning that involves artificially increasing the size and diversity of a training dataset by creating modified versions of existing data. This process helps improve model performance, prevent overfitting, and enhance robustness.
# 

# ###   Why Use Data Augmentation?
# #####   Prevents Overfitting: By introducing more varied data points, models are less likely to overfit to the training data, improving their ability to generalize to unseen data13.
# 
# #####  Improves Model Accuracy: Augmented data provides additional examples for the model to learn from, leading to better feature extraction and increased accuracy23.

# #####  Cost-Effective: Reduces the expensive costs associated with data collection, cleaning, and labeling by reusing existing data3.

# #####  Handles Class Imbalance: Helps handle skewed class distributions by oversampling minority classes

# Practicle Guide

# import the Libraies
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
import os

#import tensorflow_datasets as tfds
from tensorflow.keras import layers


# hide all warnings

import warnings
warnings.filterwarnings("ignore")
    

# download Cat Image
image_path = tf.keras.utils.get_file("cat.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")


# path of file
image_path 
# it will return the loaction of file in your system

import PIL

# This imports the Image module from Pillow, which is commonly used for image processing and manipulation tasks, including data augmentation.


# open image
PIL.Image.open(image_path)

from tensorflow.keras.utils import load_img,img_to_array

#  load_img: This function loads an image file into a PIL Image instance.
# img_to_array: This function converts a PIL Image instance into a NumPy array, 
# which is a format that can be used directly by deep learning models. 

# First way to load images
image = load_img(image_path)
# Second way to load images
load_img(image_path)


image_String1 = tf.io.read_file(image_path) 
#convert into tensor(tensor is like array but it is in binary)

# tf.io.read_file: This function reads the entire contents of a file and returns it as a string tensor.
image_String1
# binary format look like


image = tf.image.decode_jpeg(image_String1,channels=3)

# is used to decode a JPEG-encoded image into a TensorFlow tensor. 
# tf.image.decode_jpeg: This function decodes a JPEG-encoded image into a uint8 tensor.

# image_String1: This should be a string tensor containing the JPEG-encoded image data. 
# channels=3: This argument specifies that the decoded image should have 3 color channels (Red, Green, Blue).
image


plt.imshow(image)
plt.show()

# Function to visualize original and augmented images side by side.
def visualize(orignal,augmented):
    fig = plt.figure(figsize=(12,10))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(orignal)
    plt.subplot(1,2,2)
    plt.title('Augmented')
    plt.imshow(augmented)
    plt.show()
# Expalnetion 
"""
# Function to visualize original and augmented images side by side.
def visualize(original, augmented):
    Visualize original and augmented images.

    Parameters:
    - original: The original image.
    - augmented: The augmented version of the original image.
    
    # Create a figure with specified size for better visualization.
    fig = plt.figure(figsize=(12, 10))
    
    # Create a subplot for the original image.
    plt.subplot(1, 2, 1)
    
    # Set title for the original image subplot.
    plt.title('Original')
    
    # Display the original image.
    plt.imshow(original)
    
    # Create a subplot for the augmented image.
    plt.subplot(1, 2, 2)
    
    # Set title for the augmented image subplot.
    plt.title('Augmented')
    
    # Display the augmented image.
    plt.imshow(augmented)
    
    # Show the plot (you might need to add plt.show() if it's not displayed automatically).
    plt.show()
"""


# Flip the image horizontally.
flipped = tf.image.flip_left_right(image)

# Visualize the original and flipped images.
visualize(image, flipped)



# Convert the image to grayscale.

grayscaled = tf.image.rgb_to_grayscale(image)
grayscaled_squeezed = tf.squeeze(grayscaled)
visualize(image, grayscaled_squeezed)

# Squeeze the grayscale image to remove the extra dimension.
# Visualize the original and grayscale images.


plt.imshow(tf.squeeze(grayscaled),cmap='gray_r')

# Increase the saturation of the image by a factor of 8.
# Visualize the original and saturated images.

saturated = tf.image.adjust_saturation(image, 8)
visualize(image, saturated)


# Increase the brightness of the image by 0.5.
# Visualize the original and brightened images.

bright = tf.image.adjust_brightness(image, 0.5)
visualize(image, bright)



# Crop the image from its center, retaining 50% of the area.
# Visualize the original and cropped images.
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image, cropped)



# Rotate the image by 90 degrees clockwise.
# Visualize the original and rotated images.
rotated = tf.image.rot90(image)
visualize(image, rotated)

# Adjust the contrast of the image by a factor of 0.3.
contras = tf.image.adjust_contrast(image,0.3)

# Visualize the original and contrast-adjusted images.
visualize(image, contras)


# All this augmentation function we define in once
"""
    # Create a figure with specified size for better visualization.
    fig, axs = plt.subplots(1, 2, figsize=(12, 10))
    
    # Display the original image.
    axs[0].imshow(original)
    axs[0].set_title('Original')
    
    # Display the augmented image.
    axs[1].imshow(augmented)
    axs[1].set_title('Augmented')
    
    # Layout so plots do not overlap
    fig.tight_layout()
    
    # Show the plot.
    plt.show()

# Function to flip an image horizontally.
def flip_image(image):
    return tf.image.flip_left_right(image)

# Function to convert an image to grayscale.
def grayscale_image(image):
    grayscaled = tf.image.rgb_to_grayscale(image)
    return tf.squeeze(grayscaled)

# Function to increase the saturation of an image.
def saturate_image(image):
    return tf.image.adjust_saturation(image, 8)

# Function to increase the brightness of an image.
def brighten_image(image):
    image_float = tf.cast(image, tf.float32) / 255.0
    brightened = tf.image.adjust_brightness(image_float, 0.5)
    return tf.cast(brightened * 255.0, tf.uint8)

# Function to crop an image from its center.
def crop_image(image):
    return tf.image.central_crop(image, central_fraction=0.5)

# Function to rotate an image by 90 degrees.
def rotate_image(image):
    return tf.image.rot90(image)

# Function to adjust the contrast of an image.
def adjust_contrast_image(image):
    image_float = tf.cast(image, tf.float32) / 255.0
    adjusted = tf.image.adjust_contrast(image_float, 0.3)
    return tf.cast(adjusted * 255.0, tf.uint8)

# Load an image (replace 'path_to_your_image.jpg' with the actual path).
image_path = 'path_to_your_image.jpg'
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_raw, channels=3)

# Apply different augmentations and visualize.
flipped = flip_image(image)
visualize(image, flipped)

grayscaled = grayscale_image(image)
visualize(image, grayscaled)

saturated = saturate_image(image)
visualize(image, saturated)

brightened = brighten_image(image)
visualize(image, brightened)

cropped = crop_image(image)
visualize(image, cropped)

rotated = rotate_image(image)
visualize(image, rotated)

contrast_adjusted = adjust_contrast_image(image)
visualize(image, contrast_adjusted)


"""


# # Part II)

image_path

os.listdir('C:\\Users\\demog\\.keras\\datasets\\cats_and_dogs_filtered')

# or using dorect image_path

os.listdir(image_path)
base_dir = os.path.dirname(image_path)
base_dir