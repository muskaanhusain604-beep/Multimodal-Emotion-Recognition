import numpy as np
from tensorflow.keras.models import load_model
from prepare_data import load_data
from sklearn.metrics import classification_report, confusion_matrix


TEST_DIR = "dataset/test"


if __name__ == "__main__":
    print("Loading model...")
    model = load_model("emotion_model.h5")

    print("Loading test data...")
    X_test, y_test, class_names = load_data(TEST_DIR)

    print("Predicting...")
    predictions = model.predict(X_test)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
