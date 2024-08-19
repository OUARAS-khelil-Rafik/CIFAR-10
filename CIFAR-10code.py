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
train_images_dir = './train'
train_labels_path = './trainLabels.csv'

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

#-------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

# Define paths
train_images_dir = './train'
test_images_dir = './test'
train_labels_path = './trainLabels.csv'
sample_submission_path = './sampleSubmission.csv'
submission_path = './submission.csv'

# Data Augmentation and Normalization for Test Data
def create_test_datagen():
    datagen = ImageDataGenerator(
        rescale=1./255  # Normalize pixel values
    )
    return datagen

# Load the trained model
model = load_model('/kaggle/working/model.keras')

# Create Test Data Generator
test_datagen = create_test_datagen()
test_generator = test_datagen.flow_from_directory(
    directory=test_images_dir,
    target_size=(224, 224),  # Adjust size according to model
    batch_size=32,
    class_mode=None,  # No labels in test data
    shuffle=False,  # Important for correct predictions
    seed=42
)

# Predict on test data
predictions = model.predict(test_generator, verbose=1)

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Convert predictions to labels
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = [class_labels[i] for i in predicted_classes]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': [f"{i+1}" for i in range(len(predicted_labels))],
    'label': predicted_labels
})

# Save submission file
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# Load the training history from a saved file if available
# history = pd.read_csv('./history.csv')  # Assuming history is saved

# Sample test images and predictions
def plot_sample_images(test_dir, predictions, num_samples=10):
    sample_files = os.listdir(test_dir)
    sample_files = np.random.choice(sample_files, num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, file in enumerate(sample_files):
        img_path = os.path.join(test_dir, file)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize for prediction

        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        label = class_labels[predicted_class]

        plt.subplot(2, 5, i+1)
        plt.imshow(img_array[0])
        plt.title(f"Pred: {label}")
        plt.axis('off')

    plt.show()

# Plot sample images with predictions
plot_sample_images(test_images_dir, predictions)
