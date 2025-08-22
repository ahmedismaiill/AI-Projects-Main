import streamlit as st
import cv2
import numpy as np
import tempfile
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(page_title="Car License Plate Detection", page_icon="🚘", layout='wide')

st.image("ANPR-parking.png", use_container_width=True)

st.markdown("""
# Car License Plate Detection and OCR Streamlit App 🚘🚦

**This project is an interactive web application built with Streamlit that performs automatic license plate detection and optical character recognition (OCR) on video streams.**

---
## Key Features
- **License Plate Detection** using YOLO
- **Optical Character Recognition (OCR)** using PaddleOCR
- **Real-Time or File-Based Processing**
- **User-Friendly Interface** with Streamlit
- **Result Export** (cropped plates, annotated frames)
---
            
## How It Works
1. **Upload an video**.  
2. **Detection** – The YOLO model identifies the license plate’s bounding box.  
3. **OCR** – The cropped plate is passed through an OCR engine to extract characters.  
4. **Display** – Results are shown on the interface with bounding boxes and recognized text.  
5. **Export** – Users can download detection logs, cropped plate images, or annotated videos.  

---

""")


ocr = PaddleOCR(use_angle_cls=True, lang='en')
# Prepare directories and model
saved_ids = set()
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

model = YOLO("best.pt")

# Upload section
st.title("📸 Upload the Video")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('video'):
        # Video processing
        cap = cv2.VideoCapture(uploaded_file.name)
        stframe = st.empty()

        frame_count = 0
        skip_frames = 4  # Process every 3rd frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            results = model.track(frame, persist=True, verbose=False)
            r = results[0]

            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for box, track_id, conf in zip(boxes, ids, confs):
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        w = x2 - x1
                        if w <= 185:
                            continue

                        label = f"ID {int(track_id)} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0,255,0), 2)
                        # Crop and save plate (once per ID)
                        h_img, w_img, _ = frame.shape
                        x1, y1 = max(0,x1), max(0,y1)
                        x2, y2 = min(w_img,x2), min(h_img,y2)
                        plate_crop = frame[y1:y2, x1:x2]
                        if track_id not in saved_ids and plate_crop.size > 0:
                            filename = os.path.join(save_dir, f"plate_id{int(track_id)}.jpg")
                            cv2.imwrite(filename, plate_crop)
                            print(f"Saved plate image: {filename}")
                            print(ocr.ocr(filename, cls=True)[0][0][1])
                            st.sidebar.write(f"Detected Plate ID: {str(ocr.ocr(filename, cls=True)[0][0][1][0]).upper()} , with confidence {ocr.ocr(filename, cls=True)[0][0][1][1]*100:.2f}%")
                            st.sidebar.image(filename, caption=f"Plate ID {int(track_id)}", use_container_width =True)
                            saved_ids.add(track_id)


            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


