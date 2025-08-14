# Arabic Sign Language (ASL) – Quran Kareem Tracker (YOLOv8)

This project is a **YOLOv8-based Arabic Sign Language (ASL) tracker** designed to support **Quran Kareem learning** for the deaf community.  
The model detects Arabic Sign Language gestures with **99% test accuracy**, enabling real-time hand sign recognition.

---

## 📌 Project Links

- **Kaggle Notebook (Training & Evaluation)**: [https://www.kaggle.com/code/ahmedismaiil/arabic-sign-language-od-yolov8-acc-99](https://www.kaggle.com/code/ahmedismaiil/arabic-sign-language-od-yolov8-acc-99)  
- **Kaggle Model (Download Pretrained)**: [https://www.kaggle.com/models/ahmedismaiil/asl-quran-kareem-tracker](https://www.kaggle.com/models/ahmedismaiil/asl-quran-kareem-tracker)

---

## About the Dataset

### Arabic Sign Language Recognition Letters Dataset (ArSL21L)
- **Source:** Munkhjargal Gochoo  
- **Description:**  
  This dataset contains **14,202 images** of **32 different Arabic sign language letters**, collected from 50 individuals with varied backgrounds. It includes bounding box annotations for each letter. In benchmarks with object detection models, YOLOv5l achieved a COCO mAP of 0.83. Comparisons with the older ArSL2018 dataset showed that models trained on ArSL21L outperform previous ones.  
- **Dataset Split:**  
  - Training: 9,955 images  
  - Validation: 3,403 images  
  - Test: 844 images  

---

### Arabic Sign Language Dataset 2022
- **Source:** Kaggle — [Arabic Sign Language Dataset 2022](https://www.kaggle.com/datasets/ammarsayedtaha/arabic-sign-language-dataset-2022)  
- **Description:**  
  This dataset is available on Kaggle for practicing computer vision and academic research tasks related to Arabic Sign Language. While the Kaggle page does not provide detailed metadata in the summary, it is intended to support training and testing for ASL recognition systems. You can download and explore it directly from the Kaggle link.

---

## 📂 Project Structure

Arabic Sign Language Object Detection/

├── runs/ # YOLOv8 training/inference outputs

├── app.py # Streamlit app to run the tracker

├── arabic-sign-language-od-yolov8-acc-99.ipynb # Kaggle training notebook

├── best.pt # YOLOv8 trained model (test accuracy 99%)

├── requirements.txt # Dependencies for the project

├── <various images & gifs> # Media assets, visual results


---

## 🔗 Useful Links
- **Kaggle Notebook:** [Arabic Sign Language OD YOLOv8 ACC 99](https://www.kaggle.com/code/ahmedismaiil/arabic-sign-language-od-yolov8-acc-99)  
- **Trained Model:** [ASL Quran Kareem Tracker Model](https://www.kaggle.com/models/ahmedismaiil/asl-quran-kareem-tracker)

---

## 🚀 Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/yourusername/Arabic-Sign-Language-Tracker.git
cd Arabic-Sign-Language-Tracker

2️⃣ Create and activate a virtual environment (optional but recommended)
conda create -n asl-tracker python=3.11 -y
conda activate asl-tracker

3️⃣ Install dependencies
pip install -r requirements.txt

📥 Download the Model Automatically

The trained YOLOv8 model is hosted on Kaggle.
You can download it programmatically using kagglehub:

import kagglehub

# Download latest version
path = kagglehub.model_download("ahmedismaiil/asl-quran-kareem-tracker/pyTorch/default")

print("Path to model files:", path)
# Verify model exists
import os
print("best.pt exists:", os.path.exists(os.path.join(path, "best.pt")))

▶️ Run the Web App
streamlit run app.py

