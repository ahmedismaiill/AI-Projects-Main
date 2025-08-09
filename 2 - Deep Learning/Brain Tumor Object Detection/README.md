# 🧠 Brain Tumor Detection & Classification with YOLOv8

## 📌 Overview
This project uses **YOLOv8** for real-time brain tumor detection and classification from MRI scans.  
The model is trained on a **high-quality, annotated MRI dataset** and integrated into an interactive **Streamlit web application** for easy use.

---

## 📂 About the Dataset
**Dataset Name:** Brain Tumor Detection Dataset  
**Total Images:** 5,249 MRI scans (Training + Validation)  
**Annotation Format:** YOLO bounding boxes  
**Classes:**
- **0:** Glioma  
- **1:** Meningioma  
- **2:** No Tumor  
- **3:** Pituitary  

### **Data Split**
#### Training Set:
- Glioma: 1,153 images  
- Meningioma: 1,449 images  
- No Tumor: 711 images  
- Pituitary: 1,424 images  

#### Validation Set:
- Glioma: 136 images  
- Meningioma: 140 images  
- No Tumor: 100 images  
- Pituitary: 136 images  

### **Image Characteristics**
- MRI scans from **sagittal**, **axial**, and **coronal** views.
- High-quality, cleaned, and manually annotated using **LabelImg**.

**Sources:**
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)  

---

## 🚀 Model Performance (YOLOv8)
Training & validation were done using **Ultralytics YOLOv8**.

| Class       | Precision | Recall | mAP@50 | mAP@50-95 |
|-------------|-----------|--------|--------|-----------|
| **Glioma**       | 0.891     | 0.798  | 0.901  | 0.734     |
| **Meningioma**   | 0.993     | 0.915  | 0.973  | 0.825     |
| **No Tumor**     | 0.993     | 0.972  | 0.994  | 0.850     |
| **Pituitary**    | 0.928     | 0.892  | 0.954  | 0.773     |
| **Overall**      | 0.951     | 0.894  | 0.956  | 0.795     |

**Speed:** 0.8ms preprocess, 25.1ms inference per image  
**Model Accuracy:** **96% (mAP@50)**

---

## 💻 Streamlit Application
The **Streamlit web app** allows users to:
- Upload an MRI image.
- Detect and classify brain tumors into **4 categories**.
- View bounding boxes with confidence scores.
- Get tumor descriptions and example images in the sidebar.

**App Features:**
- Interactive and user-friendly interface.
- YOLOv8 real-time inference.
- Sidebar with detailed tumor information & medical context.

---
