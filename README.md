Multimodal Emotion Recognition System
A deep learning–based multimodal emotion recognition system that combines Facial Expression Recognition and Speech Emotion Recognition using weighted probabilistic fusion.

Live Demo:
👉 https://multimodal-emotion-recognition-ngzbqonmrjxa5uopz9w3on.streamlit.app/

Overview

This project predicts human emotion by analyzing:
Facial expressions from images
Emotional cues from speech audio
A fusion layer that combines both modalities
The final emotion is determined using a weighted decision strategy.

Features
Face emotion classification using CNN
Speech emotion classification using audio feature extraction + deep learning
StandardScaler applied to speech features
Weighted fusion of predictions
Deployed using Streamlit Cloud
Publicly accessible live demo

Tech Stack
Python
TensorFlow / Keras
OpenCV
Librosa
Scikit-learn
Streamlit

Model Details:
1.Face Emotion Model
CNN-based architecture
Input size: 96x96 grayscale images
Trained on facial expression dataset
Outputs probability distribution across emotions

2.Speech Emotion Model
Extracted MFCC and spectral audio features
StandardScaler normalization
Dense neural network classifier
Trained on RAVDESS dataset (24 actors)

Fusion Strategy
Final emotion is determined using weighted probability fusion:
Final Emotion = argmax( α * Face_Prediction + β * Speech_Prediction )
Where:
α and β represent modality weights

Deployment
The application is deployed on Streamlit Community Cloud.
To run locally:
pip install -r requirements.txt
streamlit run app.py

Future Improvements
Real-time webcam emotion detection
Real-time microphone input
Model optimization for faster inference

Cross-speaker generalization improvement

Transformer-based multimodal fusion
