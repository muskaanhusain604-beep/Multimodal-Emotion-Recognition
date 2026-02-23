import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = 96

def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for label, emotion in enumerate(class_names):
        emotion_path = os.path.join(data_dir, emotion)

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = to_categorical(labels)

    return images, labels, class_names


if __name__ == "__main__":
    X_train, y_train, class_names = load_data(TRAIN_DIR)
    X_test, y_test, _ = load_data(TEST_DIR)

    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)
