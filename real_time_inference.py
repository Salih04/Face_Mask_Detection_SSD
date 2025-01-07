import cv2
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = "final.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = 64

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# Load the saved LabelEncoder to get the exact class ordering
with open(LABEL_ENCODER_PATH, "rb") as f:
    encoder_for_cnn = pickle.load(f)

# The classes_ attribute is the ordered list of classes as seen in training
class_labels = encoder_for_cnn.classes_.tolist()
print("Loaded label encoder classes:", class_labels)
# Example might be: ["with_mask", "without_mask", "wrong_mask"]

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam. Exiting...")
        break

    # Convert to grayscale (because we trained on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # For each detected face, run classification
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Resize to match training input
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized / 255.0  # normalize
        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape (1, 64, 64, 1)

        # Model prediction
        predictions = model.predict(face_input, verbose=0)[0]  # shape: (3,)
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        label = class_labels[class_idx]

        # Decide color based on label
        if label == "with_mask":
            color = (0, 255, 0)   # green
        elif label == "without_mask":
            color = (0, 0, 255)   # red
        else:  # e.g., "wrong_mask"
            color = (0, 255, 255) # yellow

        # Draw bounding box & label on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

    # Show the video frame
    cv2.imshow("Real-Time Inference", frame)

    key = cv2.waitKey(1)
    if key & 0xFF in [ord('q'), 27]:  
        break

cap.release()
cv2.destroyAllWindows()
