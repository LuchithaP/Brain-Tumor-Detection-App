from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80-20 split
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(
    "dataset_jpg",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    "dataset_jpg",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
)


base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze pretrained layers

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(4, activation="softmax"),  # 4 tumor classes
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(train_gen, validation_data=val_gen, epochs=10)


model.save("brain_tumor_classifier.h5")
