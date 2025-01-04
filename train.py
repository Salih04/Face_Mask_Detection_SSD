import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle

# Parameters
IMG_SIZE = 64

# Update these paths to your local directories
ANNOTATION_DIR = "C:\\Users\\salih\\Desktop\\annotations"  # Directory containing .xml annotation files
IMAGES_DIR     = "C:\\Users\\salih\\Desktop\\images"       # Directory containing images

# -------------------------------------------------------------------
# Function to load XML annotations
# -------------------------------------------------------------------
def load_xml_annotations(annotation_dir):
    annotations = []
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(annotation_dir, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            item = {
                'filename': root.find('filename').text,
                'annotations': []
            }

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    annotation = {
                        'class': obj.find('name').text,  # with_mask, without_mask, or wrong_mask
                        'x': float(bbox.find('xmin').text),
                        'y': float(bbox.find('ymin').text),
                        'width': float(bbox.find('xmax').text) - float(bbox.find('xmin').text),
                        'height': float(bbox.find('ymax').text) - float(bbox.find('ymin').text)
                    }
                    item['annotations'].append(annotation)

            annotations.append(item)
    return annotations

# -------------------------------------------------------------------
# Preprocess Images
# -------------------------------------------------------------------
def preprocess_images(annotations, img_size=(64, 64)):
    images, labels = [], []

    for item in annotations:
        file_path = os.path.join(IMAGES_DIR, item['filename'])
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
            continue

        for bbox in item.get('annotations', []):
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
            resized_img = cv2.resize(cropped_img, img_size)
            # Convert to grayscale (since we'll train on grayscale)
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            images.append(gray_img)
            labels.append(bbox['class'])

    return np.array(images), labels

# -------------------------------------------------------------------
# Load annotations and preprocess images
# -------------------------------------------------------------------
annotations = load_xml_annotations(ANNOTATION_DIR)
X, y = preprocess_images(annotations, img_size=(IMG_SIZE, IMG_SIZE))

# Normalize and reshape
X = X / 255.0  # Normalize pixel values to [0, 1]
X_cnn = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # For CNN input

# -------------------------------------------------------------------
# Encode labels
# -------------------------------------------------------------------
encoder_for_cnn = LabelEncoder()
encoded_y_cnn = encoder_for_cnn.fit_transform(y)
y_cnn = to_categorical(encoded_y_cnn, num_classes=3)  # 3 classes: with_mask, without_mask, wrong_mask

# -------------------------------------------------------------------
# (Optional) Train and evaluate SVM (on flattened images)
# -------------------------------------------------------------------
def train_svm(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # Flatten images
    X_flat = X.reshape(len(X), -1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

    # Train SVM
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return svm_model, label_encoder

svm_model, label_encoder_svm = train_svm(X, y)

# -------------------------------------------------------------------
# (Optional) Visualize SVM predictions
# -------------------------------------------------------------------
def visualize_svm_predictions(svm_model, X_test, y_test, label_encoder, num_samples=10):
    y_pred = svm_model.predict(X_test)
    class_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

    indices = np.random.choice(len(X_test), num_samples, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        true_label = class_labels[y_test[idx]]
        pred_label = class_labels[y_pred[idx]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    plt.show()

# For visualization, we need a separate train_test_split for X_svm_flat
X_svm_flat = X.reshape(len(X), -1)
y_svm_encoded = LabelEncoder().fit_transform(y)
_, X_test_svm, _, y_test_svm = train_test_split(X_svm_flat, y_svm_encoded, test_size=0.2, random_state=42)

visualize_svm_predictions(svm_model, X_test_svm, y_test_svm, LabelEncoder().fit(y), num_samples=10)

# -------------------------------------------------------------------
# Train CNN
# -------------------------------------------------------------------
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train-test split for CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

cnn_model = create_cnn_model((IMG_SIZE, IMG_SIZE, 1))
cnn_model.fit(
    X_train_cnn,
    y_train_cnn,
    epochs=80,
    validation_data=(X_test_cnn, y_test_cnn),
    batch_size=32
)

# Evaluate CNN
loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------------------------------
# Visualize CNN predictions
# -------------------------------------------------------------------
def visualize_predictions(model, X_test, y_test, label_encoder, num_samples=10):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    class_labels = label_encoder.inverse_transform(range(3))
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        plt.axis('off')

        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    plt.show()

visualize_predictions(cnn_model, X_test_cnn, y_test_cnn, encoder_for_cnn, num_samples=10)

# -------------------------------------------------------------------
# Save the final Keras model
# -------------------------------------------------------------------
cnn_model.save("final.keras")
print("CNN model saved as 'final.keras")

# -------------------------------------------------------------------
# Save the LabelEncoder used for the CNN
# -------------------------------------------------------------------
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder_for_cnn, f)
print("LabelEncoder saved as 'label_encoder.pkl'.")
