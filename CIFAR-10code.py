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
