import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from src.audio_features import extract_features

# Load models
face_model = load_model("emotion_model.h5")
speech_model = load_model("speech_emotion_model.h5")
scaler = joblib.load("speech_scaler.pkl")

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

st.title("Multimodal Emotion Recognition System")
st.write("Upload an image and an audio file to predict emotion.")

# Upload image
image_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

# Upload audio
audio_file = st.file_uploader("Upload Audio File", type=["wav"])

if image_file and audio_file:
    # ---------- FACE ----------
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = np.reshape(img, (1, 96, 96, 3))

    face_prediction = face_model.predict(img)
    face_index = np.argmax(face_prediction)
    face_emotion = emotion_labels[face_index]

    # ---------- SPEECH ----------
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    features = extract_features("temp.wav")
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    speech_prediction = speech_model.predict(features)
    speech_index = np.argmax(speech_prediction)
    speech_emotion = emotion_labels[speech_index]

    # ---------- FUSION ----------
    min_len = min(face_prediction.shape[1], speech_prediction.shape[1])

    face_probs = face_prediction[0][:min_len]
    speech_probs = speech_prediction[0][:min_len]

    final_probs = 0.6 * face_probs + 0.4 * speech_probs
    final_index = np.argmax(final_probs)
    final_emotion = emotion_labels[final_index]

    # ---------- DISPLAY ----------
    st.subheader("Results")
    st.write("Face Emotion:", face_emotion)
    st.write("Speech Emotion:", speech_emotion)
    st.success(f"Final Combined Emotion: {final_emotion}")