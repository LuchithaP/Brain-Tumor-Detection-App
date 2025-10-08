import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_classifier.h5")

# Load the image you want to predict
img_path = "test\meningioma\me-0296.jpg"
IMG_SIZE = 224  # size used during training

# Load and resize image
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)

# Preprocess
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

# If it's categorical classification
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class index:", predicted_class)

# If you have class labels
class_labels = [
    "meningioma",
    "glioma",
    "pituitary",
    "no tumour",
]
print("Predicted label:", class_labels[predicted_class[0]])
print("Prediction confidence:", np.max(predictions))
print("Raw model output:", predictions)
