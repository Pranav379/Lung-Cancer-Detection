# 🫁 Deep Learning Lung Cancer Classifier

## 🚀 Project Overview
This project is a deep learning-based web application that classifies lung CT scan images into multiple cancer categories using a fine-tuned Swin Transformer model.

Users can upload a CT scan image and receive real-time predictions with confidence scores across five classes:
- Adenocarcinoma  
- Large Cell Carcinoma  
- Squamous Cell Carcinoma  
- Malignant
- Normal / Benign  

---

## 🌐 Live Demo
👉 https://lung-ct-classifier.streamlit.app

---

## 🧠 Model
- **Architecture:** Swin Transformer
- **Framework:** PyTorch  
- **Input:** 224 × 224 CT scan images  
- **Output:** 5-class classification  

### Preprocessing
- Resize to 224×224  
- Convert grayscale → 3-channel  
- Normalize using ImageNet mean and standard deviation  

---

## 🖥️ Features
- Upload CT scan images (JPG, PNG, TIFF, etc.)  
- Real-time predictions with confidence scores

---

## 📦 Model
The trained model is hosted externally on HuggingFace and is automatically downloaded at runtime using Hugging Face Hub.

---

## 📊 Dataset
CT scan dataset sourced from: https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer

---

## 📌 Future Work
- Improve model accuracy with larger datasets
- Add Grad-CAM visualizations for interpretability
- Optimize deployment with GPU-backed inference
