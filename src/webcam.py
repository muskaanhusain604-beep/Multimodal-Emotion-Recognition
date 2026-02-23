import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 96

# Load model
model = load_model("emotion_model.h5")

# Emotion labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emotion_history = []

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        current_emotion = class_names[class_index]

        # Add to history
        emotion_history.append(current_emotion)

        # Keep only last 10 predictions
        emotion_history = emotion_history[-10:]

        # Stabilize prediction using majority vote
        emotion = max(set(emotion_history), key=emotion_history.count)

        label = f"{emotion} ({confidence:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
