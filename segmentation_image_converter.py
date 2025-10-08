import h5py
import numpy as np
import cv2
import os

input_folder = "dataset"
image_folder = "segmentation/images"
mask_folder = "segmentation/masks"

os.makedirs(image_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".mat"):
        file_path = os.path.join(input_folder, file_name)
        with h5py.File(file_path, "r") as f:
            cjdata = f["cjdata"]
            image = np.array(cjdata["image"]).T
            mask = np.array(cjdata["tumorMask"]).T

            # Normalize MRI image
            image_norm = (
                (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            ).astype(np.uint8)
            mask_bin = (mask > 0).astype(np.uint8) * 255

            base_name = os.path.splitext(file_name)[0]
            cv2.imwrite(os.path.join(image_folder, f"{base_name}.png"), image_norm)
            cv2.imwrite(os.path.join(mask_folder, f"{base_name}.png"), mask_bin)
