import os
import cv2
import matplotlib.pyplot as plt

TRAIN_DIR = "dataset/train"

def show_sample_images():
    emotion = os.listdir(TRAIN_DIR)[0]
    emotion_path = os.path.join(TRAIN_DIR, emotion)

    images = os.listdir(emotion_path)[:5]

    plt.figure(figsize=(10, 4))
    for i, img_name in enumerate(images):
        img_path = os.path.join(emotion_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.suptitle(f"Sample images from class: {emotion}")
    plt.show()

if __name__ == "__main__":
    show_sample_images()

