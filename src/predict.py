import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 96

# Load trained model
model = load_model("emotion_model.h5")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicted Emotion: {class_names[class_index]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_image(image_path)
