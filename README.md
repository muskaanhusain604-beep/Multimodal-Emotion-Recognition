Multimodal Emotion Recognition System

## Demo

![App Demo](app_demo.png)

# Multimodal Emotion Recognition System

## Overview
This project implements a **Deep Learning–based Multimodal Emotion Recognition System** that combines facial expressions and speech signals to predict human emotions.

The system integrates predictions from two modalities:

- Facial Expression Recognition (image-based)
- Speech Emotion Recognition (audio-based)
- Weighted probabilistic fusion for final decision making

By analyzing both **visual and vocal cues**, the system improves reliability compared to traditional single-modality emotion recognition models.

---

## Problem Statement
Emotion recognition using only a single modality (face or speech) can often be unreliable due to various environmental and behavioral factors.

Common challenges include:

- Lighting variations affecting facial detection
- Background noise in audio recordings
- Ambiguous facial expressions
- Tone inconsistencies in speech

This project addresses these limitations by **combining predictions from both modalities**, producing a more robust and reliable emotion classification.

---

## System Architecture

The system processes two different inputs and merges their predictions.

### Image Processing Pipeline
Image Input → CNN Face Emotion Model → Emotion Probability Distribution

### Audio Processing Pipeline
Audio Input → Feature Extraction (MFCC + Spectral Features) → Dense Neural Network → Emotion Probability Distribution

### Fusion Layer
The outputs from both models are combined using **weighted probability fusion** to produce the final emotion prediction.

---

## Tech Stack

The system is implemented using the following technologies:

- Python
- TensorFlow / Keras
- OpenCV
- Librosa
- Scikit-learn
- Streamlit

---

## Model Details

### Face Emotion Model

The facial emotion recognition model is based on a **Convolutional Neural Network (CNN)**.

Key characteristics:

- CNN-based architecture
- Input size: **96 × 96 grayscale images**
- Pixel values normalized
- Outputs probability distribution across **7 emotion classes**

---

### Speech Emotion Model

The speech emotion recognition model analyzes acoustic features extracted from audio signals.

Dataset used:

- **RAVDESS dataset**
- 24 actors with multiple emotional speech recordings

Model pipeline:

- Feature extraction using **MFCC and spectral features**
- Data normalization using **StandardScaler**
- Dense neural network classifier
- Outputs probability distribution across emotions

---

## Preprocessing Steps

### Image Preprocessing

Before feeding images into the CNN model:

- Resize image to **96 × 96**
- Normalize pixel values
- Reshape image to match model input dimensions

---

### Audio Preprocessing

For speech input:

- Extract **MFCC features**
- Extract **spectral features**
- Apply the same **StandardScaler used during training**
- Reshape feature vector for model input

Ensuring consistent preprocessing between training and inference is critical for accurate predictions.

---

## Fusion Strategy

The final emotion prediction is computed using **weighted probability fusion**.

```
Final Emotion = argmax(0.6 × Face_Prediction + 0.4 × Speech_Prediction)
```

### Why Weighted Fusion?

- Facial expressions are generally **more visually stable**
- Speech signals can be affected by **background noise**
- Weighted fusion improves the **overall reliability of predictions**

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

To run the application locally:

```
pip install -r requirements.txt
streamlit run app.py
```

This launches an interactive web interface for testing emotion predictions.

---

## Key Challenges Solved

During development several practical challenges were addressed:

- Handling **tensor shape mismatches**
- Maintaining consistent preprocessing during inference
- Reusing the same **StandardScaler** used during model training
- Aligning emotion labels between image and audio models
- Deploying deep learning models on **Streamlit Cloud**
- Cleaning and structuring the Git repository for production

---

## Future Improvements

Potential improvements for the system include:

- Real-time **webcam and microphone integration**
- Transformer-based **multimodal fusion models**
- Model optimization for **faster inference**
- Improved **cross-speaker generalization**
- Performance evaluation dashboard with detailed metrics

---

## Conclusion

This project demonstrates the practical implementation of **multimodal deep learning**, combining computer vision and speech processing techniques.

It showcases the ability to:

- Design and train deep learning models
- Integrate multiple modalities
- Debug and optimize model pipelines
- Deploy AI applications using cloud platforms

The system represents a complete **end-to-end AI solution for emotion recognition**.
