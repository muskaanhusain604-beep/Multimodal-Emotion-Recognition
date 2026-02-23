from tensorflow.keras.models import load_model

# Load face model
face_model = load_model("emotion_model.h5")

# Load speech model
speech_model = load_model("speech_emotion_model.h5")

print("Both models loaded successfully!")

import numpy as np
import cv2

# Dummy test image (just to test pipeline)
img = cv2.imread("Dataset/test/happy/PrivateTest_218533.jpg")  # change to any test image you have
img = cv2.resize(img, (96, 96))
img = img / 255.0
img = np.reshape(img, (1, 96, 96, 3))

face_prediction = face_model.predict(img)
print("Face prediction:", face_prediction)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

predicted_index = np.argmax(face_prediction)
predicted_emotion = emotion_labels[predicted_index]

print("Face Emotion:", predicted_emotion)

from audio_features import extract_features
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
scaler = joblib.load("speech_scaler.pkl")

audio_path = "audio/sample.wav"  # change if needed

features = extract_features(audio_path)
features = np.array(features)

# reshape
features = features.reshape(1, -1)

# APPLY SAME SCALER
features = scaler.transform(features)

speech_prediction = speech_model.predict(features)

print("Speech prediction:", speech_prediction)

speech_emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

speech_index = np.argmax(speech_prediction)
speech_emotion = speech_emotion_labels[speech_index]

print("Speech Emotion:", speech_emotion)

# Ensure both predictions have same length
min_len = min(face_prediction.shape[1], speech_prediction.shape[1])

face_probs = face_prediction[0][:min_len]
speech_probs = speech_prediction[0][:min_len]

# Weighted fusion
final_probs = 0.6 * face_probs + 0.4 * speech_probs

final_index = np.argmax(final_probs)

final_emotion = speech_emotion_labels[final_index]  # same label order assumed

print("Final Combined Emotion:", final_emotion)