#!/usr/bin/env python
# coding: utf-8

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Hide Warnings
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow_hub as hub


# LOADING DATASET

###  Constants
Data_dir = "V:\\UTKFace"



# Convert Data Dir to Path Object
data_dir = os.path.join(Data_dir)


# List all entries (files and directories) in the folder
entries = os.listdir(Data_dir)
print(entries)

# choose dir crop_part1
data_dir_crop = os.path.join(data_dir,'crop_part1/')
data_dir_crop

# Initialize empty lists to store images, ages, and genders
images = []
ages = []
genders = []

# Loop over the first 8000 filenames in the directory
for i in os.listdir(data_dir_crop)[:8000]:
    
    # Each filename is expected to be in the format: <age>_<gender>_<race>_<date&time>.jpg
    # Example: "25_0_1_20170109181326335.jpg.chip.jpg"
    split = i.split('_')
    
    # Extract age (first part of filename) and convert to integer
    ages.append(int(split[0]))
    
    # Extract gender (second part of filename) and convert to integer
    # Typically, 0 = male, 1 = female in UTKFace dataset
    genders.append(int(split[1]))
    
    # Open the image file and append the PIL Image object to the images list
    images.append(Image.open(data_dir_crop + i))

images
len(images)
ages
len(ages)
genders
len(genders)

# Convert the list of image objects into a pandas Series named 'Images'
images = pd.Series(list(images), name='Images')

# Convert the list of ages into a pandas Series named 'Ages'
ages = pd.Series(list(ages), name='Ages')

# Convert the list of genders into a pandas Series named 'Genders'
genders = pd.Series(list(genders), name='Genders')

# Concatenate the three Series into a single DataFrame along the columns (axis=1)
# This creates a structured table with images, ages, and genders as columns
df = pd.concat([images, ages, genders], axis=1)

# Display the resulting DataFrame
df



# display  demo images from images with ages and Genders

display(df['Images'][115])
print(df['Ages'][115],
      df['Genders'][115])

# 0 = female 
# 1 = male



display(df['Images'][5050])
print(df['Ages'][5050],
      df['Genders'][5050])


# plot the distribution to see dataset

# Set the default Seaborn theme for better aesthetics in plots
sns.set_theme()

sns.distplot(df['Ages'], kde=True, bins=30)


# Plot the distribution of the 'Ages' column from the DataFrame
# - kde=True adds a Kernel Density Estimate curve to smooth the histogram
# - bins=30 divides the age range into 30 intervals for the histogram bars
# plt.show()  # Uncomment this line if running outside of Jupyter Notebook to display the plot

# Too many faces of people between 0 and 4 years old. 
# The model would fit too well to these ages and not enough to the other ages. 
# To resolve this I'm only going to include a third of the images between these ages.


# Select rows where age <= 4 and randomly sample 30% of them
under4age = df[df['Ages'] <= 4].sample(frac=0.3)

# Select rows where age > 4 (all of them)
over4age = df[df['Ages'] > 4]

# Concatenate the sampled under4age rows with all over4age rows
# ignore_index=True resets the index in the combined DataFrame
df = pd.concat([over4age, under4age], ignore_index=True)

# Verify the changes by plotting the age distribution again
df['Ages']

# plot the distribution
sns.distplot(df['Ages'],kde=True, bins=30)


# show ages greater then 80 
df = df[df['Ages'] < 80]


# plot the distribution
sns.distplot(df['Ages'],kde=True, bins=20)

# plot the Counterplot 
plt.figure(figsize=(8,6))
sns.countplot(x='Genders', data=df, palette='pastel')
plt.xticks(ticks=[0, 1, 2], labels=['Female', 'Male', 'Trans'])
plt.xlabel('Gender')
plt.ylabel('Number of Images')
plt.title('Distribution of Images by Gender')
plt.show()


# Not sure what 3 corresponds to - both genders, no gender, unknown, 
# or just an error in the data entry?
# To be safe, I am going to remove any rows where gender equals 3.


df = df[df['Genders'] != 3]
# Check the number of unique gender classes
num_gender_classes = df['Genders'].nunique()
print(f"Number of unique gender classes: {num_gender_classes}")

# Check unique gender values
unique_genders = df['Genders'].unique()
print("Unique gender values:", unique_genders)

# Count the occurrences of each gender in the dataset       
gender_counts = df['Genders'].value_counts()
print(gender_counts)

# ## Image Resizing and Array Conversion
# Initialize empty lists to store processed image arrays and labels
x = []
y = []

# Loop through each row in the DataFrame
for i in range(len(df)):
    # Resize each image to 200x200 pixels using ANTIALIAS filter for better quality
    # Note: ANTIALIAS is deprecated in newer Pillow versions, consider using Image.LANCZOS instead
    
   # df['Images'].iloc[i] = df['Images'].iloc[i].resize((200,200), Image.ANTIALIAS)
    df['Images'].iloc[i] = df['Images'].iloc[i].resize((200,200), Image.LANCZOS)
    
    # Convert the PIL Image object to a NumPy array for model processing
    ar = np.asarray(df['Images'].iloc[i])
    
    # Add the image array to our features list
    x.append(ar)
    
    # Create a label array containing [age, gender] for each image
    # Convert to integers to ensure proper data type for model training
    
    agegen = [int(df['Ages'].iloc[i]), int(df['Genders'].iloc[i])]
    
    # Add the label array to our labels list
    y.append(agegen)

# Convert the list of image arrays to a NumPy array for efficient processing
# This creates a 4D array with shape (n_samples, height, width, channels)
x = np.array(x)
x


# ## Train Test Split
# splitting  dataset into training and testing sets for two different targets: age and gender.


# ### Normalize value before spliting

# Normalize pixel values to [0,1]
x = x.astype('float32') / 255.0

# Split the dataset into training and testing sets for age and  gender prediction
# Using stratified sampling to maintain class distribution in both sets 
y_age = df['Ages']
y_gender = df['Genders']

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender)

#Check for Class Imbalance
print(y_gender.value_counts())
print(y_age.value_counts())
