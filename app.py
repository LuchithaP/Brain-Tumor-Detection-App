import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import h5py
import base64

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
CLASS_IMG_SIZE = 224
CLASS_MODEL_PATH = "brain_tumor_classifier.h5"
SEG_MODEL_PATH = "tumor_segmentation_model.h5"
CLASS_NAMES = [
    "meningioma",
    "glioma",
    "pituitary",
    "no tumour",
]


# -----------------------------
# Background image
# -----------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


background_path = "bg.jpg"
if os.path.exists(background_path):
    set_background(background_path)


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    class_model = tf.keras.models.load_model(CLASS_MODEL_PATH)
    seg_model = tf.keras.models.load_model(SEG_MODEL_PATH)
    return class_model, seg_model


class_model, seg_model = load_models()

# -----------------------------
# Containers for Title and Description
# -----------------------------
st.markdown(
    """
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; box-shadow: 2px 2px 8px #ccc;">
        <h1 style="text-align:center; color:#4B0082;">🧠 Brain Tumor Detection WebApp</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background-color:#e8eaf6; padding:15px; border-radius:10px; margin-top:10px;">
        <p style="text-align:center; font-size:16px; color:#333;">
        Upload a brain MRI image to automatically detect the type of brain tumor 
        (Meningioma, Glioma, Pituitary) and highlight the tumor area if present.
        Images are displayed in separate boxes for clarity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a brain MRI image (.jpg, .png, or .mat)", type=["jpg", "png", "mat"]
)


# -----------------------------
# Helper functions
# -----------------------------
def preprocess_classification(img):
    if len(img.shape) == 2:
        img_resized = cv2.resize(img, (CLASS_IMG_SIZE, CLASS_IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img_resized = cv2.resize(img[:, :, 0], (CLASS_IMG_SIZE, CLASS_IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.resize(img, (CLASS_IMG_SIZE, CLASS_IMG_SIZE))
    img_rgb = img_rgb / 255.0
    img_rgb = img_rgb.astype(np.float32)
    return np.expand_dims(img_rgb, axis=0)


def preprocess_segmentation(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img_resized, axis=(0, -1))


def predict_segmentation(img_input):
    pred_mask = seg_model.predict(img_input)[0, :, :, 0]
    return (pred_mask > 0.5).astype(np.uint8) * 255


def overlay_mask(original_img, mask):
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    if len(img_resized.shape) == 2:
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_resized
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    return overlay


def load_mat_image(file):
    f = h5py.File(file, "r")
    cjdata = f["cjdata"]
    img = np.array(cjdata["image"]).T
    img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(
        np.uint8
    )
    return img_norm


# -----------------------------
# Process uploaded image
# -----------------------------
if uploaded_file is not None:
    if uploaded_file.name.endswith(".mat"):
        img = load_mat_image(uploaded_file)
    else:
        img = np.array(Image.open(uploaded_file).convert("L"))

    # Classification
    class_input = preprocess_classification(img)
    preds = class_model.predict(class_input)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100
    tumor_type = CLASS_NAMES[class_index]

    # Container for prediction
    st.markdown(
        f"""
        <div style="background-color:#fff3e0; padding:15px; border-radius:10px; margin-top:10px;">
            <h3 style="text-align:center; color:#ff5722;">Predicted Tumor Type: {tumor_type} ({confidence:.2f}%)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Segmentation overlay if tumor exists
    if tumor_type != "No Tumor":
        seg_input = preprocess_segmentation(img)
        mask = predict_segmentation(seg_input)
        overlay_img = overlay_mask(img, mask)
    else:
        overlay_img = None

    # Display images side by side in separate containers
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background-color:#e0f7fa; padding:10px; border-radius:10px; margin-bottom:10px;">
                <h4 style="text-align:center; color:#00796b;">Original MRI Image</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(img, use_column_width=True, clamp=True, channels="L")

    with col2:
        st.markdown(
            """
            <div style="background-color:#ffebee; padding:10px; border-radius:10px; margin-bottom:10px;">
                <h4 style="text-align:center; color:#c62828;">Tumor Overlay</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if overlay_img is not None:
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), use_column_width=True
            )
        else:
            st.info("No tumor detected.")
