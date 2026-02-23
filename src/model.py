from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

def build_model():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)
    )

    base_model.trainable = True

# Freeze first 100 layers
    for layer in base_model.layers[:100]:
     layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model

def compile_model(model):
    model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
    return model
