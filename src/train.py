from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import build_model, compile_model

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = 96
BATCH_SIZE = 64


if __name__ == "__main__":

    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Build model
    model = build_model()
    model = compile_model(model)

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train
    model.fit(
        train_generator,
        epochs=15,
        validation_data=test_generator,
        callbacks=[early_stop]
    )

    model.save("emotion_model.h5")
