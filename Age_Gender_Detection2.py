#!/usr/bin/env python
# coding: utf-8

# Age and Gender Detection


# Enable mixed precision globally BEFORE any model or layer creation
# pip install protobuf==3.20.*

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')  # Use mixed precision (float16 + float32)

# import libraries
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

    



# Constants and Data Preparation
Data_dir = "V:\\UTKFace"


# Convert Data Dir to Path Object
data_dir = os.path.join(Data_dir)



# List all entries (files and directories) in the folder
entries = os.listdir(Data_dir)
print(entries)

# choose dir crop_part1
data_dir_crop = os.path.join(data_dir,'crop_part1')
data_dir_crop



# Constants
IMAGE_SIZE = 240 # EfficientNetB4 is designed for 380x380 images, 
BATCH_SIZE = 32


# Get the list of filenames (e.g., first 8000 files)
# Parse filenames for age/gender

filenames = [f for f in os.listdir(data_dir_crop) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:8000]
ages, genders, valid_filenames = [], [], []
for fname in filenames:
    try:
        split = fname.split('_')
        age = int(split[0])
        gender = int(split[1])
        ages.append(age)
        genders.append(gender)
        valid_filenames.append(fname)
    except Exception as e:
        continue # Skip files with parsing errors
        


df = pd.DataFrame({
    'filename': valid_filenames,
    'Ages': ages,
    'Genders': genders
})
print(f"Dataset size after cleaning: {df.shape}")


# Prepare Dataset DataFrame
# Strip whitespace from filenames (precaution)
# Clean: remove outliers, missing files, invalid genders

df = df[df['Genders'].isin([0, 1])]
df = df[df['Ages'] < 80]
df = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(data_dir_crop, x)))]
# Check for missing files and remove them

df['Ages'] = df['Ages'] / 80.0   # Normalize ages to [0,1]
df

# Stratified split by age bins and gender for balanced validation

df['age_bin'] = pd.cut(df['Ages'], bins=8, labels=False)
train_df, val_df = train_test_split(
    df, test_size=0.2, 
    stratify=df[['Genders', 'age_bin']], 
    random_state=42
)
train_df = train_df.drop(columns=['age_bin'])
val_df = val_df.drop(columns=['age_bin'])


df.head()



# ###  Data Augmentation


# Custom preprocessing function for augmentation

def custom_preprocess(image):
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return preprocess_input(image) # EfficientNet preprocessing

# Data generators with augmentation for training and simple preprocessing for validation

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    fill_mode='nearest'
)


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir_crop,
    x_col='filename',
    y_col=['Ages', 'Genders'],
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=True
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=data_dir_crop,
    x_col='filename',
    y_col=['Ages', 'Genders'],
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# print gender, to see if any imbalanced data 
print("Gender distribution:\n", df['Genders'].value_counts())

# Custom generator to split labels for multi-output model
def multi_output_generator(generator):
    while True:
        x, y = next(generator)
        age_labels = y[:, 0].reshape(-1, 1)     # shape (batch_size, 1)
        gender_labels = y[:, 1].reshape(-1, 1)  # shape (batch_size, 1)
        yield x, {'age': age_labels, 'gender': gender_labels}

train_gen = multi_output_generator(train_generator)
val_gen = multi_output_generator(val_generator)


# ## Use EfficientNetB4 Multi-Task Model

# Model definition
# Model definition with mixed precision


from tensorflow.keras.applications.efficientnet import EfficientNetB4


def build_model(IMAGE_SIZE):
    base_model = EfficientNetB4(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # dtype float32 by default
    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)  # runs in float32 automatically
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layers explicitly set to float32 for numeric stability
    age_output = Dense(1, activation='sigmoid', name='age', dtype='float32')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=[age_output, gender_output])
    return model, base_model




# Compile and train initial model

model, base_model = build_model(IMAGE_SIZE)
optimizer = Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss={'age': Huber(delta=1.0), 'gender': 'binary_crossentropy'},
    loss_weights={'age': 0.6, 'gender': 0.4},
    metrics={'age': 'mae', 'gender': 'accuracy'}
)

model.summary()


# ## === Callbacks (monitor correct metric) ===


# Train the Model with EarlyStopping and Checkpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

def cosine_annealing(epoch):
    lr_min, lr_max, T_max = 1e-6, 1e-4, 30
    lr = lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return float(lr)  # Ensure it's a Python float


early_stop = EarlyStopping(monitor='val_age_mae', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint('improved_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=True)
lr_scheduler = LearningRateScheduler(cosine_annealing)




# In[22]:


# If you want to save the history as JSON, convert all values to lists or floats
# Utility to sanitize history for JSON

def sanitize_history_fixed(history):
    def convert(v):
        if isinstance(v, tf.Tensor):
            return convert(v.numpy())
        elif isinstance(v, np.ndarray):
            return [convert(x) for x in v]
        elif isinstance(v, (np.generic, np.number)):
            return v.item()
        elif isinstance(v, (np.float32, np.float64)):
            return float(v)
        elif isinstance(v, list):
            return [convert(x) for x in v]
        elif isinstance(v, dict):
            return {k: convert(val) for k, val in v.items()}
        else:
            return v
    return {k: convert(v) for k, v in history.items()}



# # Training 

# Initial training with frozen base
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

# After initial training
with open('history.json', 'w') as f:
    json.dump(sanitize_history_fixed(history.history), f)

# Save the model weights after initial training
model.save_weights('improved_model.keras')


# Save the  model 
# model.save('train_age_gender_model.keras')


# # Fine-tuning:


# fine tunning
IMAGE_SIZ = 240

# 1. Rebuild the model architecture EXACTLY as before
model, base_model = build_model(IMAGE_SIZ)  # Use your build_model() function or full model definition

# 2. Load the saved weights from initial training
model.load_weights('improved_model.keras')

# 3. Unfreeze the last N layers of the base model (e.g., last 50)
for layer in base_model.layers[-50:]:
    layer.trainable = True

# (Optional but recommended) Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

# 4. Re-compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss={'age': Huber(delta=1.0), 'gender': 'binary_crossentropy'},
    loss_weights={'age': 0.6, 'gender': 0.4},
    metrics={'age': 'mae', 'gender': 'accuracy'}
)

# 5. Fine-tune the model
history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,  # You can adjust the number of epochs
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stop, checkpoint, lr_scheduler]
)


# 6. Save history safely
with open('history_finetune.json', 'w') as f:
    json.dump(sanitize_history_fixed(history_finetune.history), f)

# 7. Save the fine-tuned model weights
model.save_weights('final_age_gender_weights.h5')

# Save the full model (architecture + weights)
# model.save('final_age_gender_model.h5')



## Prediction Function 


# manually select image
# Prediction Function with TTA & Calibration

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError
import tkinter as tk
from tkinter import filedialog

# --- Calibration Function ---
def calibrate_age(raw_age):
    # Make this configurable if you want to tune later
    if raw_age < 0.25:
        return raw_age * 1.15
    elif raw_age > 0.75:
        return raw_age * 0.92
    return raw_age

# --- Prediction Function with TTA, Error Handling, and Optional TTA ---
def predict_age_gender(image_path, model, img_size=240, tta_rounds=5, use_tta=True, verbose=False):
    """
    Predicts age and gender from an image file using the model.
    Includes test-time augmentation (TTA) and error handling.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except (FileNotFoundError, UnidentifiedImageError) as e:
        if verbose:
            print(f"Error loading image {image_path}: {e}")
        return None, None, None

    ages, genders = [], []
    rounds = tta_rounds if use_tta else 1
    for _ in range(rounds):
        aug_img = img.copy()
        # Add more augmentations if desired, but keep consistent with training
        if use_tta and np.random.rand() > 0.5:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        aug_img = ImageOps.fit(aug_img, (img_size, img_size), Image.LANCZOS)
        img_array = np.asarray(aug_img)
        if img_array.shape[-1] != 3:
            if verbose:
                print(f"Image {image_path} does not have 3 channels.")
            return None, None, None
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        try:
            age_pred, gender_pred = model.predict(img_array, verbose=0)
        except Exception as e:
            if verbose:
                print(f"Prediction error for {image_path}: {e}")
            return None, None, None
        ages.append(age_pred[0][0])
        genders.append(gender_pred[0][0])
    age = np.mean(ages)
    gender_prob = np.mean(genders)
    age = calibrate_age(age)
    age = np.clip(age, 0, 1)  # Clip to [0,1] before denormalizing
    age = int(round(age * 80))
    gender = 'female' if gender_prob > 0.5 else 'male'
    confidence = gender_prob if gender == 'female' else 1 - gender_prob
    return age, gender, confidence


# #### Load train model

# --- Example Usage ---
from tensorflow.keras.models import load_model

# Load your trained model or weights before using the functions above
# Example:
model, base_model = build_model(IMAGE_SIZE)
model.load_weights('final_age_gender_weights.h5')

# or load the full model if saved as such


# #### Single Image Prediction Display

# --- Single Image Prediction with Robustness ---

def show_prediction(image_path, model, img_size=240, use_tta=True, verbose=True):
    age, gender, confidence = predict_age_gender(
        image_path, model, img_size=img_size, use_tta=use_tta, verbose=verbose
    )
    if age is None:
        if verbose:
            print(f"Prediction failed for {image_path}.")
        return None, None, None
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{os.path.basename(image_path)}")
        plt.show()
    except Exception as e:
        if verbose:
            print(f"Could not display image {image_path}: {e}")
    print(f"Predicted Age: {age}")
    print(f"Predicted Gender: {gender} ({confidence*100:.2f}%)")
    return age, gender, confidence



# single image 

image_dir = r"V:\celebrity images"
image_filename = "ranbir.jpeg"
image_path = os.path.join(image_dir, image_filename)

if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
#    img = Image.open(image_path)
#    plt.imshow(img)
#    plt.axis('off')
#    plt.title(os.path.basename(image_path))  # This will display 'ranbir.jpg'
#   plt.show()

# age, gender, confidence = predict_age_gender(image_path, model)
# print(f"Predicted Age: {age}")
# print(f"Predicted Gender: {gender} ({confidence*100:.2f}%)")

    show_prediction(image_path, model, img_size=IMAGE_SIZE, use_tta=True)


# single image 

image_dir = r"V:\celebrity images"
image_filename = "pulkit.jpeg"
image_path = os.path.join(image_dir, image_filename)

if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
#    img = Image.open(image_path)
#    plt.imshow(img)
#    plt.axis('off')
#    plt.title(os.path.basename(image_path))  # This will display 'ranbir.jpg'
#   plt.show()

# age, gender, confidence = predict_age_gender(image_path, model)
# print(f"Predicted Age: {age}")
# print(f"Predicted Gender: {gender} ({confidence*100:.2f}%)")

    show_prediction(image_path, model, img_size=IMAGE_SIZE, use_tta=True)


# #### GUI Batch Prediction
# 
# ##### we create a function which Automaticaaly take a input as a image


# --- GUI: Select and Predict on Multiple Images ---
# we create a function which Automaticaaly take a input as a image

def select_and_predict(model, img_size=240, use_tta=True, verbose=True):
    """
    Opens a file dialog to select image(s), makes predictions, and displays results.
    """
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title='Select image(s)',
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    for image_path in file_paths:
        try:
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Input Image")
            plt.show()
        except Exception as e:
            if verbose:
                print(f"Could not display image {image_path}: {e}")
            continue
        age, gender, confidence = predict_age_gender(
            image_path, model, img_size=img_size, use_tta=use_tta, verbose=verbose
        )
        if age is None:
            print(f"Prediction failed for {image_path}.")
        else:
            print(f"Image: {image_path}")
            print(f"Predicted Age: {age}")
            print(f"Predicted Gender: {gender} ({confidence*100:.2f}%)\n")


# For GUI batch prediction:
select_and_predict(model, img_size=IMAGE_SIZE, use_tta=True)


# . Automatically Process All Images in a Folder
# This will process all images in a given directory.


def predict_on_folder(folder_path, model, img_size=200):
    """
    Predicts age and gender for all images in a folder and displays each image with its prediction.
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        # Predict (your existing function)
        age, gender, confidence = predict_age_gender(image_path, model, img_size=img_size)
        # Display image and prediction
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{filename}\nAge: {age} | Gender: {gender} ({confidence*100:.1f}%)")
        plt.show()  # This will pause and display each image until the window is closed

from tensorflow.keras.models import load_model
model = load_model('final_age_gender_model.keras')
predict_on_folder(r"V:\celebrity images", model, img_size=380)


tf.keras.backend.clear_session()

# --- GUI: Select and Predict on Multiple Images ---
import tkinter as tk
from tkinter import filedialog

def select_and_predict(model, img_size=240):
    """
    Opens a file dialog to select image(s), makes predictions, and displays results.
    """
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title='Select image(s)',
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    for image_path in file_paths:
        show_prediction(image_path, model, img_size=img_size)

# Uncomment to use GUI:
select_and_predict(model, img_size=IMAGE_SIZE)

# Uncomment to use GUI:
select_and_predict(model, img_size=IMAGE_SIZE)

# --- Predict on All Images in a Folder ---
def predict_on_folder(folder_path, model, img_size=240):
    """
    Predicts age and gender for all images in a folder and displays each image with its prediction.
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        show_prediction(image_path, model, img_size=img_size)

# Example usage:
predict_on_folder(r"V:\celebrity images", model, img_size=IMAGE_SIZE)