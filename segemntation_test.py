import tensorflow as tf
import numpy as np
import cv2
import os


IMG_SIZE = 224  # Must match training size
SEG_MODEL_PATH = "tumor_segmentation_model.h5"


model = tf.keras.models.load_model(SEG_MODEL_PATH)
print("Segmentation model loaded successfully!")


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0, -1))  # (1, IMG_SIZE, IMG_SIZE, 1)
    return img, img_input  # return original image and preprocessed


def predict_mask(img_input):
    pred_mask = model.predict(img_input)[0, :, :, 0]  # Remove batch and channel dims
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    return pred_mask_bin


def overlay_mask(original_img, mask):
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    return overlay


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import cv2

    # -----------------------------
    # Enter your image path here
    # -----------------------------
    image_path = "test\healthy\healthy1.jpg"  # <-- change this to your image file

    # Load and preprocess image
    original_img, img_input = load_image(image_path)

    # Predict tumor mask
    pred_mask = predict_mask(img_input)

    # Overlay mask on original image
    overlay_img = overlay_mask(original_img, pred_mask)

    # Save overlay image
    output_path = os.path.splitext(image_path)[0] + "_overlay.png"
    cv2.imwrite(output_path, overlay_img)
    print(f"Tumor overlay saved to: {output_path}")

    # Display overlay using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Tumor Overlay")
    plt.show()
