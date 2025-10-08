import h5py
import numpy as np
import cv2
import os

input_folder = "dataset"
output_folder = "dataset_jpg"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".mat"):
        file_path = os.path.join(input_folder, file_name)

        with h5py.File(file_path, "r") as f:
            cjdata = f["cjdata"]

            # Access fields
            image = np.array(cjdata["image"]).T  # transpose may be needed
            label = int(np.array(cjdata["label"])[0, 0])

            # Normalize image
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = image.astype(np.uint8)

            # Save based on label
            label_folder = os.path.join(output_folder, str(label))
            os.makedirs(label_folder, exist_ok=True)

            file_base = os.path.splitext(file_name)[0]
            cv2.imwrite(os.path.join(label_folder, f"{file_base}.jpg"), image)
