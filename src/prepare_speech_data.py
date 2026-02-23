import os
import numpy as np
from audio_features import extract_features
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "speech_dataset"

# RAVDESS emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

emotion_list = list(emotion_map.values())


def load_speech_data():
    features = []
    labels = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Extract emotion code from filename
                # RAVDESS filename format: 03-01-05-01-02-02-12.wav
                emotion_code = file.split("-")[2]

                if emotion_code in emotion_map:
                    emotion = emotion_map[emotion_code]

                    mfcc_features = extract_features(file_path)

                    if mfcc_features is not None:
                        features.append(mfcc_features)
                        labels.append(emotion_list.index(emotion))

    X = np.array(features)
    y = to_categorical(labels)

    return X, y, emotion_list


if __name__ == "__main__":
    X, y, emotions = load_speech_data()

    print("Speech Data Shape:", X.shape)
    print("Labels Shape:", y.shape)
    print("Emotions:", emotions)
