#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install split-folders


# In[ ]:





# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Num GPUs Available: ", len(gpus))
else:
    print("No GPU detected.")


# In[1]:


import pathlib
import splitfolders
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os


# In[3]:


import pathlib
import splitfolders

Dataset_dir = r'..\..\Dataset\Pistachio_Image_Dataset'

Dataset_dir = pathlib.Path(Dataset_dir)

# Ensure that splitfolders is imported
# If not, you can import it using:
# from splitfolders import split

# Use the split function to create the output folders
splitfolders.ratio(Dataset_dir, output="../../Dataset/Pistachio_Dataset", seed=101, ratio=(.7, .2, .1))


# In[ ]:


## Shape of the image


# In[ ]:





# In[4]:


# We can check image shape using this code
import cv2
import numpy as np

# Load the image
image = cv2.imread('../../Dataset/Pistachio_Dataset/train/Kirmizi_Pistachio/kirmizi (10).jpg')

# Check the shape of the image
height, width, channels = image.shape
# Dataset comprises of images of shape: (600,600,3)
# Print the image shape
print(f"Image shape: (height={height}, width={width}, channels={channels})")


# In[ ]:





# In[5]:


training_dir = "../../Dataset//Pistachio_dataset/train"
test_dir = "../../Dataset//Pistachio_dataset/test"
val_dir = "../../Dataset//Pistachio_dataset/val"


# ##
# Data augmentation is a technique used to artificially increase the diversity of your training dataset by applying various transformations to the existing images. This helps improve the generalization and robustness of a machine learning model. Let's go through each parameter in your code:

# In[6]:


# Define image size and batch size
image_height, image_width = 224, 224
batch_size = 32

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation Data Generator (No Data Augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)


# ### Load and Preprocess Data:
# * Load your dataset and preprocess it. 
# * If you are working with image data, you might use the ImageDataGenerator for data augmentation and preprocessing.

# In[7]:


# Training Data Generator
# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['Kirmizi_Pistachio', 'Siirt_Pistachio']  # Specify class labels
)

# Validation Data Generator

val_datagen = ImageDataGenerator(
    rescale=1./255
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['Kirmizi_Pistachio', 'Siirt_Pistachio']  # Specify class labels
)


# In[8]:


# We have to created image generator for the validation dataset


# For testing

test_datagen = ImageDataGenerator(
    rescale=1./255
    )
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True,
    classes=['Kirmizi_Pistachio', 'Siirt_Pistachio']  # Specify class labels
)


# In[10]:


import pandas as pd
import glob

Total_TrainImages = glob.glob('../../Dataset/Pistachio_dataset/train/*/*.jpg')
print("Total number of training images: ", len(Total_TrainImages))


# In[21]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of VGG16
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # Use Global Average Pooling instead of Flatten
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Assuming 2 classes: Kirmizi_Pistachio and Siirt_Pistachio

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
num_epochs= 60
# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Assuming you have already trained the model and stored the history
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]  # Include EarlyStopping callback
)


# In[ ]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")


# In[ ]:


import matplotlib.pyplot as plt

# Assuming you have already trained the model and stored the history
#history = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)

# Plot training and validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




