import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(image_dir, csv_file):
    images = []
    labels = []

    # Load CSV file with image paths and blood groups
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image'])
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            continue
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if image is None:
            print(f"Error: {image_path} could not be loaded. Skipping.")
            continue
        
        image = cv2.resize(image, (224, 224))  # Resize image to 224x224
        image = image / 255.0  # Normalize pixel values to [0, 1]

        images.append(image)
        labels.append(row['blood_group'])  # Blood group as label

    images = np.array(images)
    labels = np.array(labels)

    # Split data into train and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = load_and_preprocess_data('dataset_blood_group/', 'dataset.csv')

# Optional: Print the shape of the output to verify
print(f"Training images shape: {X_train.shape}")
print(f"Test images shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")
