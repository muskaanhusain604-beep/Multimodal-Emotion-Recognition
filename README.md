Multimodal Emotion Recognition System

## Demo

![App Demo](app_demo.png)

Overview:

This project implements a deep learning–based Multimodal Emotion Recognition System that combines:

Facial Expression Recognition (Image-based)

Speech Emotion Recognition (Audio-based)

Weighted probabilistic fusion for final decision making

The system predicts human emotions by analyzing both visual and vocal cues, improving robustness compared to single-modality systems.


Problem Statement:

Emotion recognition using a single modality (only face or only speech) can be unreliable due to:

Lighting variations

Background noise

Ambiguous facial expressions

Tone inconsistencies

This project addresses that limitation by combining predictions from both modalities to produce a more reliable final output.


System Architecture:

Image Input → CNN-based Face Model → Emotion Probabilities

Audio Input → Feature Extraction (MFCC + Spectral Features) → Dense Neural Network → Emotion Probabilities

Fusion Layer → Weighted Average → Final Emotion


Tech Stack:

Python

TensorFlow / Keras

OpenCV

Librosa

Scikit-learn

Streamlit

Model Details:

Face Emotion Model

CNN-based architecture

Input size: 96x96 images

Normalized pixel values

Outputs probability distribution across 7 emotions


Speech Emotion Model:

Trained on RAVDESS dataset (24 actors)

Audio feature extraction using MFCC and spectral features

StandardScaler normalization applied

Dense neural network classifier

Outputs probability distribution across emotions

Preprocessing Steps:

*Image:

Resize to 96x96

Normalize pixel values

Reshape to match model input

*Audio:

Extract MFCC and spectral features

Apply StandardScaler (same scaler used during training)

Reshape to match model input


Fusion Strategy:

Final emotion is computed using weighted probability fusion:

Final Emotion = argmax( 0.6 × Face_Prediction + 0.4 × Speech_Prediction )

Why weighted fusion?

Facial expressions are generally more stable visually

Speech can be affected by noise

Weighted fusion improves decision reliability


Deployment

The application is deployed using Streamlit Community Cloud.


To run locally:

pip install -r requirements.txt

streamlit run app.py

Key Challenges Solved


Handling tensor shape mismatches:

Ensuring consistent preprocessing during inference

Reusing the same StandardScaler used during training

Managing multimodal label alignment

Deploying deep learning models on Streamlit Cloud

Cleaning Git repository for production


Future Improvements:

Real-time webcam and microphone integration

Transformer-based multimodal fusion

Model optimization for faster inference

Cross-speaker generalization improvement

Performance evaluation metrics dashboard


Conclusion:

This project demonstrates practical implementation of multimodal deep learning, model integration, and cloud deployment. It showcases the ability to build, debug, and deploy an end-to-end AI system.

