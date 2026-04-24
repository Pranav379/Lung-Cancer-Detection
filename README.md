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

## ⚙️ Setup & Run Locally
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
- Probability breakdown across all classes  
- Clean and interactive UI built with Streamlit
