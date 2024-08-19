import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define paths
train_labels_path = './trainLabels.csv'
train_images_dir = './train'

# Load train labels
train_labels = pd.read_csv(train_labels_path)

# Display first 10 images (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Iterate over the first 10 images
for i, ax in enumerate(axes.flat):
    img_name = train_labels.iloc[i]['id']  # Get image id from CSV
    img_label = train_labels.iloc[i]['label']  # Get corresponding label
    img_path = os.path.join(train_images_dir, f"{img_name}.png")  # Image file path

    # Load and display the image
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(f"{img_label}")
    ax.axis('off')  # Hide the axes

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_images_dir = './train'
test_images_dir = './test'

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images in the range (degrees)
    width_shift_range=0.2,  # Randomly translate images horizontally
    height_shift_range=0.2,  # Randomly translate images vertically
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Fill pixels with the nearest value
    validation_split=0.2  # Split data for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for test data

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_images_dir,
    target_size=(32, 32),  # Resize images to match model input size
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    train_images_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_images_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode=None,  # No labels for test data
    shuffle=False
)

print("Data loading and preprocessing complete.")