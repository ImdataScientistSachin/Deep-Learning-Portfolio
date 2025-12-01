#!/usr/bin/env python
# coding: utf-8


#  Deploy Deep Learning Model using OpenCV

# Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import warnings

# Hide Warnings
warnings.filterwarnings("ignore")
import os
import tensorflow_hub as hub



# Constants

Data_dir = "C:\\Users\\demog\\.keras\\datasets\\ML_MODEL"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 6
EPOCHS = 15



#image_path =  tf.keras.utils.get_file('C:\\Users\\demog\\.keras\\datasets\\ML_MODEL')
# confirm if dir exist in location

if not os.path.exists(Data_dir):
    print("Folder does not exist.")
else:
    print(f"Folder exists at: {Data_dir}")



# Convert Data Dir to Path Object
data_dir = os.path.join(Data_dir)
data_dir



# List all files and directories directly inside data_dir

sachin_images = os.path.join(data_dir,'sachin')
shubhangi_images = os.path.join(data_dir,'shubhangi')

# Count number of images in each folder
num_sachin_img = len(os.listdir(sachin_images))
num_shubhangi_img = len(os.listdir(shubhangi_images))

print("Numbers of Sachin Images: ",num_sachin_img)
print("Numbers of shubhangi Images: ",num_shubhangi_img)

# Total number of images
total_images = num_sachin_img+num_shubhangi_img
print("Total Numbers of Images: ",total_images)



import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


# print one of the sample Image using PIL (pillow) .
s_image = load_img('C:/Users/demog/.keras/datasets/ML_MODEL/sachin/sac-5_flipped.jpg')
s_image
Sh_image = load_img('C:/Users/demog/.keras/datasets/ML_MODEL/shubhangi/shubh-9_cropped.jpg')
Sh_image



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

# find class name
class_names = train_ds.class_names
print(class_names)


# Extract a sample image and its corresponding label from the training dataset
# - next(iter(train_ds)): Get the first batch of images and labels from the dataset


sample_img, labels = next(iter(train_ds))
sample_img
labels


# plot the image

plt.imshow(sample_img[1].numpy().astype("uint8"))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.show()

# plot multiple images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")



# configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# prepare model with pretrain model
import keras
base_model = keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), 
                                                 include_top=False, 
                                                 pooling='avg')

base_model.summary()


# prepare model don't train on default configuration layer
base_model.trainable = False



# prepare custom layer for pretrain model 

from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Model
x = Dense(1024, 'relu')(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, 'relu')(x)

out = Dense(1, 'sigmoid')(x)
model = Model(inputs=base_model.input, outputs=out)
model.summary()

# compile model 
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

# train the model
initial_epochs = EPOCHS
history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

# evaluate the model
loss, accuracy = model.evaluate(val_ds)
print("Loss: ",loss)
print("Accuracy: ",accuracy)

# plot the training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

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


img =  load_img('C:/Users/demog/.keras/datasets/ML_MODEL/sachin/sac-1_randomly_rotated.jpg')
img



# resize images

test = img.resize((IMG_HEIGHT,IMG_WIDTH))
test

#  Convert the resized image to a NumPy array
test_array = np.array(test)

# convert to array & reshape 

test_array = test_array.reshape(1,224,224,3)
test_array.shape

# test prediction
prediction = model.predict(test_array)
prediction


# GET OUT the arrays
prediction[0][0].round()

# convert it to integer
pred = int(prediction[0][0].round())
pred



# print class name
class_names[(pred)]


# prediction 2

img2 =  load_img('C:/Users/demog/.keras/datasets/ML_MODEL/shubhangi/shubh-1_randomly_rotated.jpg')
img2

#  convert to array & reshape 
test1 = img2.resize((IMG_HEIGHT,IMG_WIDTH))
test1

# Convert the resized image to a NumPy array
test_array = np.array(test1)
test_array2 = test_array.reshape(1,224,224,3)
test_array2.shape


# test prediction
prediction1 = model.predict(test_array2)
prediction1



# GET OUT the arrays
prediction1[0][0].round()


# convert it to integer
pred2 = int(prediction1[0][0].round())
pred2



# print class name
class_names[(pred2)]


# ## Use Cascade Classifier for identify Faces
class_names = ['sachin','shubhangi']

#testing cascad
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
while True:
     # Read the frame from the webcam
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test = img
    test = cv2.resize(test,(224,224))
    #test = test/255.0
    test = test.reshape(1,224,224,3)
    #iden = []
    
    # Predict using the model
    pred = model.predict(test)
    #print(pred[0][0])
    #print(iden)
    

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5)
    #print(faces.shape)
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
        cv2.putText(img, str(class_names[int(pred[0][0].round())]),
                    (x+5,y-5),
                    font, 1, (0,165,255), 3)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()


# # Cascade Classifier Function for Capture Images

import cv2
import glob
import os

# function for user input their name so the snap save on their name.
# Prompt user for their name
user_name = input("Please enter your name: ").strip()

# Check if the name is provided
if not user_name:
    print("No name provided. Exiting...")
    cv2.destroyAllWindows()
    exit()  # Exit if no name is provided

# Create directory for saving images if it doesn't exist
directory = f"CV_data/{user_name}"
if not os.path.exists(directory):
    os.makedirs(directory)


cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        x1 = x+5
        y1 = y+5
        x2 = x+w
        y2 = y+h
        # Crop the face
        cropped_face = img[y1:y2, x1:x2]

        # Resize the cropped face to 224x224
        resized_face = cv2.resize(cropped_face, (256, 256))

        # Save the captured image into the datasets folder
        
        # Save the captured image into the user's directory
        cv2.imwrite(f"{directory}/{user_name}_{count}.jpg", resized_face)
        count += 1
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)        
        cv2.imshow('image', img)
        
            
        
    k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
    if k == 50:
        break
    elif count >= 100: # Take 100 face sample and stop video
         break

print(f"Captured {count} images.")
cam.release()
cv2.destroyAllWindows()