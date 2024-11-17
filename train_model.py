import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os

# Define the label mapping
label_mapping = {
    'A+': 0,
    'A-': 1,
    'B+': 2,
    'B-': 3,
    'AB+': 4,
    'AB-': 5,
    'O+': 6,
    'O-': 7
}

# 1. Load and Preprocess the Image Data
def load_and_preprocess_images(image_dir, img_size=(224, 224)):
    images = []
    labels = []
    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_mapping[label])
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0
    images = np.expand_dims(images, axis=-1)
    return images, labels

# Load the images and labels
image_dir = 'D:/Github projects/Blood-group-prediction/dataset_blood_group'
X, y = load_and_preprocess_images(image_dir)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(label_mapping), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Train the Model
model = create_model()
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# 6. Save the Model
#model.save('fingerprint_blood_group_model.h5')
#print("Model saved as 'fingerprint_blood_group_model.h5'")

model.save('my_model.keras')
print("Model saved as 'my_model.keras'")
