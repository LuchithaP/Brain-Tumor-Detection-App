from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


IMG_SIZE = 224


def load_data(img_folder, mask_folder):
    images = sorted(os.listdir(img_folder))
    masks = sorted(os.listdir(mask_folder))
    X, Y = [], []
    for img_file, mask_file in zip(images, masks):
        img = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
        X.append(np.expand_dims(img, -1))
        Y.append(np.expand_dims(mask, -1))
    return np.array(X), np.array(Y)


X, Y = load_data("segmentation/images", "segmentation/masks")


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# U-Net model
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    # Decoder
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(c6)

    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(16, 3, activation="relu", padding="same")(c7)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)
    model = models.Model(inputs, outputs)
    return model


model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=8)

model.save("tumor_segmentation_model.h5")
