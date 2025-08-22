# Car License Plate Detection and OCR Streamlit App 🚘🚦

An interactive **web application** built with **Streamlit** that performs **automatic license plate detection** using YOLO and **optical character recognition (OCR)** using PaddleOCR. The app supports both **real-time video input** and **file uploads**, providing cropped plate images, detected text, and confidence scores in an easy-to-use interface.

---

## ✨ Features

- **License Plate Detection** – powered by YOLO
- **OCR Recognition** – using PaddleOCR
- **Real-Time or Uploaded Video Processing**
- **Streamlit Interface** – clean and interactive
- **Result Export** – save cropped plates and detection logs automatically
- **Confidence Filtering** – only reliable detections are shown

---

## 📂 Project Structure

├── best.pt # Trained YOLO model weights

├── ANPR-parking.png # App banner image

├── app.py # Main Streamlit application

├── results/ # Saved cropped license plate images

├── requirements.txt # Python dependencies

└── README.md # Project documentation


## Install dependencies

pip install -r requirements.txt

<img width="1862" height="1059" alt="Screenshot 2025-08-22 030516" src="https://github.com/user-attachments/assets/6909be7c-70de-43e0-b1ac-545003665085" />

<img width="1866" height="1059" alt="Screenshot 2025-08-22 030528" src="https://github.com/user-attachments/assets/ecf95a13-d3a1-4692-878a-857197cef4d1" />

<img width="1849" height="1005" alt="Screenshot 2025-08-22 030543" src="https://github.com/user-attachments/assets/57e5cc97-561a-46b1-9a28-b50c02d964f1" />
