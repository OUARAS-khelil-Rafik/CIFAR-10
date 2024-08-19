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

import os
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

# Define paths
train_images_dir = './train'  # Update if using a different path
train_labels_path = './trainLabels.csv'  # Update with the correct path

# Function to organize images into class directories
def organize_images(base_dir, labels_file):
    labels = pd.read_csv(labels_file)
    for _, row in labels.iterrows():
        class_dir = os.path.join(base_dir, row['label'])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        img_file = os.path.join(base_dir, f"{row['id']}.png")
        if os.path.exists(img_file):
            shutil.move(img_file, os.path.join(class_dir, f"{row['id']}.png"))

# Organize images into directories
organize_images(train_images_dir, train_labels_path)

# Verify directory structure
print("Directory structure:")
for root, dirs, files in os.walk(train_images_dir):
    if dirs:
        for d in dirs:
            print(f"Class directory {d} contains {len(os.listdir(os.path.join(root, d)))} images")

# Set up MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Data Augmentation and Normalization
def create_datagen():
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        rotation_range=20,  # Data Augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

# Create Data Generators
datagen = create_datagen()
train_generator = datagen.flow_from_directory(
    directory=train_images_dir,
    target_size=(224, 224),  # Adjust size according to model
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Define and compile the model within the strategy scope
with strategy.scope():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
try:
    model.fit(
        train_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size
    )
except Exception as e:
    print(f"Error during training: {e}")

# Save the trained model
model.save('./model/model.keras')  # Save in Keras format
print("Model saved to ./model/model.keras")
