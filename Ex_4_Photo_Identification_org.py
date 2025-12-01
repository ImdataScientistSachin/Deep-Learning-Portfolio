#!/usr/bin/env python
# coding: utf-8

"""  # Apply FACE Identification using Deep Learning Model """

# ##  Step 1 - Apply Agumentation on all directories (Because the data is bit short)


"""
## Apply Augmentation on sachin dir

# Directory containing resized images


resized_dir = 'C:/Users/demog/.keras/datasets/ML_MODEL/sachin/resized_images'
output_dir = os.path.join('C:/Users/demog/.keras/datasets/ML_MODEL/sachin', "augmented_images")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Augmentation functions
def flip_image(image):
    return tf.image.flip_left_right(image)

def grayscale_image(image):
    grayscaled = tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(grayscaled)  # Convert back to RGB

def saturate_image(image):
    return tf.image.adjust_saturation(image, 8)

def brighten_image(image):
    image_float = tf.cast(image, tf.float32) / 255.0
    brightened = tf.image.adjust_brightness(image_float, 0.5)
    return tf.cast(brightened * 255.0, tf.uint8)

def crop_image(image):
    return tf.image.central_crop(image, central_fraction=0.5)

def rotate_image(image):
    return tf.image.rot90(image)

def adjust_contrast_image(image):
    image_float = tf.cast(image, tf.float32) / 255.0
    adjusted = tf.image.adjust_contrast(image_float, 0.3)
    return tf.cast(adjusted * 255.0, tf.uint8)

def random_rotate_image(image):
    angle = tf.random.uniform([], minval=0, maxval=360, dtype=tf.int32)
    return tf.image.rot90(image, k=angle // 90)

# Dictionary of augmentations
augmentations = {
    "flipped": flip_image,
    "grayscale": grayscale_image,
    "saturated": saturate_image,
    "brightened": brighten_image,
    "cropped": crop_image,
    "rotated": rotate_image,
    "contrast_adjusted": adjust_contrast_image,
    "randomly_rotated": random_rotate_image,
}



"""


# ## step 2 -  Resizing all image so low memory consuption


"""
# Process each image in the resized directory
for filename in os.listdir(resized_dir):
    if filename.endswith(".jpg"):  # Only process JPEG files
        file_path = os.path.join(resized_dir, filename)
        
        try:
            # Load and decode the image
            image_raw = tf.io.read_file(file_path)
            image = tf.io.decode_jpeg(image_raw, channels=3)  # Ensure RGB format
            
            # Apply each augmentation and save results
            for aug_name, aug_func in augmentations.items():
                augmented_image = aug_func(image)
                
                # Ensure the augmented image has the correct shape [height, width, 3]
                if len(augmented_image.shape) == 4:  # If an extra batch dimension exists
                    augmented_image = tf.squeeze(augmented_image, axis=0)
                
                # Save augmented image
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{aug_name}.jpg")
                augmented_encoded = tf.io.encode_jpeg(tf.cast(augmented_image, tf.uint8))
                tf.io.write_file(output_path, augmented_encoded)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

"""



# ## Step 3 - apply Deep Learning model


# Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import warnings

# Hide Warnings
warnings.filterwarnings("ignore")


# Constants

Data_dir = "C:\\Users\\demog\\.keras\\datasets\\ML_MODEL"
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 5
EPOCHS = 15

#image_path =  tf.keras.utils.get_file('C:\\Users\\demog\\.keras\\datasets\\ML_MODEL')
# confirm if dir exist in location

if not os.path.exists(Data_dir):
    print("Folder does not exist.")
else:
    print(f"Folder exists at: {Data_dir}")


# Convert Data Dir to Path Object

data_dir = pathlib.Path(Data_dir).with_suffix('')


# count the number of JPEG images within subdirectories of the data_dir directory. 

import glob
# Count JPEG images in subdirectories

jpeg_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
image_count = sum(
    len(list(data_dir.glob(f'**/*{ext}'))) 
    for ext in jpeg_extensions
)

print(f'The number of JPEG images is: {image_count}')



# List all files and directories directly inside data_dir

items = list(data_dir.glob('*'))
print('Items directly inside data_dir:')
for item in items:
    print(item)

# check one of the class from dir
(list(data_dir.glob('sachin')))

# check one of the class from dir
(list(data_dir.glob('shubhangi')))

# sac_dir = 'C:/Users/demog/.keras/datasets/ML_MODEL/sachin'
# shubh_dir ='C:/Users/demog/.keras/datasets/ML_MODEL/shubhangi'

# Print contents and size of sachin directory

sachin_dir = data_dir / 'sachin'
print(f"Size of sachin directory: {len(list(sachin_dir.glob('*')))}")
print("\nContents of sachin directory:")
print(list(sachin_dir.glob('*')))


# print shubhangi directory

shubh_dir = data_dir / 'shubhangi'
print(f"Size of sachin directory: {len(list(shubh_dir.glob('*')))}")
print("\nContents of shubhangi directory:")
print(list(shubh_dir.glob('*')))


# print one of the sample Image using PIL (pillow) .
S_image = PIL.Image.open('C:/Users/demog/.keras/datasets/ML_MODEL/sachin/augmented_images/sac-3_flipped.jpg')
S_image


# print one of the sample Image using
from tensorflow.keras.utils import load_img,img_to_array


# load shubhangi images
Sh_image = load_img('C:/Users/demog/.keras/datasets/ML_MODEL/shubhangi/augmented_images/shubh-9_cropped.jpg')
Sh_image

# Convert to numpy array
sh_img_array = img_to_array(Sh_image)   
# prepare training dataset

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
)


# Create a validation dataset from the images in data_dir
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
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
)



# Get Class Names
class_names = train_ds.class_names
print("Class names:", class_names)
print(f"Class 0: {class_names[0]}")
print(f"Class 1: {class_names[1]}")



# Define AUTOTUNE for Prefetching
AUTOTUNE = tf.data.AUTOTUNE

# Explore the Data
# Extract a sample image and its corresponding label from the training dataset
# - next(iter(train_ds)): Get the first batch of images and labels from the dataset
sample_img, labels = next(iter(train_ds))

# Convert the sample image tensor to a NumPy array and ensure its data type is uint8
sample_img.numpy().astype('uint8')
print (labels)

# print one the Image
plt.imshow(sample_img[3].numpy().astype('uint8'))
print(labels[3])
print(class_names[0])



# ploting SubPlot 
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    # Ensure you only iterate up to the number of images available
    for i in range(min(len(images), 9)):  # Use min() to avoid out-of-bounds access
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# Define AUTOTUNE for dynamic buffer size in prefetching

# Cache and shuffle the training dataset, then prefetch with dynamic buffer size
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Cache and prefetch the validation dataset with dynamic buffer size
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# prepare model
model = Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),  # Added another layer
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Increased dense layer size
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

# Compile Model without  using the 'Softmax' Activation
from tensorflow.keras.optimizers import Adam

model.compile(optimizer= Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# train model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Plot Training and Validation Metrics
acc = history.history['accuracy']  # Training accuracy at each epoch
val_acc = history.history['val_accuracy']  # Validation accuracy at each epoch

loss = history.history['loss']  # Training loss at each epoch
val_loss = history.history['val_loss']  # Validation loss at each epoch

# Define the range of epochs for plotting
epochs_range = range(EPOCHS)

# Create a figure with two subplots
plt.figure(figsize=(8, 4))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='center')  # Position legend at lower right corner


# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Display the plot
plt.show()

# Test the Model on New Images
# modified image with custom dimention
img = tf.keras.utils.load_img('sac_test2.jpg', target_size=(IMG_HEIGHT, IMG_WIDTH))
img

# convert img  to numpy array
# Load Test Image
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
print('predictions value ',predictions[0][0])


# Load Test Images
test_img = tf.keras.utils.load_img('sac_test2.jpg', target_size=(IMG_HEIGHT, IMG_WIDTH))
test_img1 = tf.keras.utils.load_img('shubh_test1.jpg', target_size=(IMG_HEIGHT, IMG_WIDTH))

# Display Test Images
test_img
test_img1


# Convert to NumPy Arrays
test_img_array = tf.keras.utils.img_to_array(test_img)
test_img_array = test_img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)


# shubhu image reshaping
test_img_array1 = tf.keras.utils.img_to_array(test_img1)
test_img_array1 = test_img_array1.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)



"""
# Make Prediction
predictions = model.predict(test_img_array)
probability1 = predictions[0][0]

if probability1 > 0.5:
    predicted_class1 = class_names[1]
else:
    predicted_class1 = class_names[0]

print(f"Predicted Class for Sachin Image: {predicted_class1}")
"""

# Make Prediction 1 (sachin)
predictions = model.predict(test_img_array)
probability1 = predictions[0][0]
print(f"prediction value for Sachin image: {probability1}")

# Verify which class corresponds to which index
if probability1 > 0.5:
    predicted_class1 = class_names[0]  # If probability > 0.5, predict sachin
else:
    predicted_class1 = class_names[1]  # If probability < 0.5, predict shubhangi

print(f"Predicted Class for Sachin Image: {predicted_class1}")


# Make Prediction 2
predictions2 = model.predict(test_img_array1)
probability2 = predictions2[0][0]
print(f" prediction value for Shubhangi image: {probability2}")

# For Shubhangi's image
if probability2 > 0.5:
    predicted_class2 = class_names[1]  # shubhangi
else:
    predicted_class2 = class_names[0]  # sachin
print(f"Predicted Class for Shubhangi Image: {predicted_class2}")
