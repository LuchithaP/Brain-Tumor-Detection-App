# 🧠 Brain Tumor Detection & Segmentation App

This project is a **deep learning-based web application** designed to **detect and highlight brain tumors** in MRI images. It uses a combination of **image classification** and **segmentation models** to both identify the presence of a tumor and visually highlight its location.

---

## 🚀 Features

- Upload MRI images for analysis  
- Classify images as **Tumor** or **No Tumor**  
- Segment and highlight the **tumor region** on the image  
- Built using **TensorFlow**, **OpenCV**, and **Streamlit**  
- Easy-to-use web interface  

---

## 🧩 Model Overview

The app consists of two main models:
1. **Classification Model** – Detects whether a brain MRI image shows a tumor.  
2. **Segmentation Model** – Identifies and highlights the tumor region using pixel-level analysis.  

Both models are trained using the Figshare Brain Tumor Dataset.

---

## 📚 Dataset

This project uses the **Brain Tumor Dataset** from Figshare.

- **Source:** [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)  
- **Description:** The dataset contains MRI brain images categorized into tumor and non-tumor classes, suitable for both classification and segmentation tasks.  

**Citation:**
> Brain Tumor Dataset. Figshare.  
> URL: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427  

Please acknowledge the dataset authors if you use this dataset in your work.

---

## ⚙️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
