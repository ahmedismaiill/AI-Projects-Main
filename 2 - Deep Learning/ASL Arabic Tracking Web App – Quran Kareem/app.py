import streamlit as st
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from ultralytics import YOLO

arabic_map = {
    "aleff": "ا",
    "bb": "ب",
    "ta": "ت",
    "thaa": "ث",
    "jeem": "ج",
    "haa": "ح",
    "khaa": "خ",
    "dal": "د",
    "thal": "ذ",
    "ra": "ر",
    "zay": "ز",
    "seen": "س",
    "sheen": "ش",
    "saad": "ص",
    "dhad": "ض",
    "taa": "ط",
    "dha": "ظ",
    "ain": "ع",
    "ghain": "غ",
    "fa": "ف",
    "gaaf": "ق",
    "kaaf": "ك",
    "laam": "لام",
    "la": "لا",
    "meem": "ميم",
    "nun": "ن",
    "ha": "ه",
    "waw": "و",
    "ya": "ي",
    "yaa": "ي",
    "toot": "ة",
    "al": "ال"
}


st.cache_data.clear()  
st.cache_resource.clear() 

st.set_page_config(page_title="ASL Arabic Tracking", page_icon="🤖",layout='wide')
st.markdown("# ASL Arabic Tracking Web App 👱🏻 – Quran Kareem 📖")

st.image("23bc0c218792587.67a74fe84f292.png", use_container_width =True)

st.markdown("""
This application uses **YOLO (You Only Look Once)** for **real-time tracking of Arabic Sign Language (ASL) letters** through your webcam.

---

## Project Introduction
This typeface represents an innovative achievement that combines visual art and modern technology to serve the deaf community and enhance their ability to learn and understand the Holy Quran. Specifically designed to support the Arabic Sign Language alphabet, this typeface expresses its letters through simplified and clear hand movement illustrations. It has three different styles, making it versatile and easy to use in various educational and Quranic applications.

### Features of the Typeface:
1. **Arabic Sign Language Alphabet:**  
   The typeface is based on the Arabic Sign Language alphabet, where hand movements representing letters are transformed into simplified visual forms. This process was carried out in collaboration with experts from the Sign Language Association to ensure accuracy and clarity of expressions.

2. **Three Design Styles:**  
   The typeface is available in three styles to meet user needs in various contexts, whether for long texts, headings, or special designs.

3. **Full Arabic Language Support:**  
   Supports all forms of Arabic letters in their four states (isolated, initial, medial, and final), ensuring users face no issues when using it for writing Quranic or educational texts.

4. **Balanced Geometric Design:**  
   The typeface’s geometry ensures consistent proportions and sizes of hand movements (fingers, wrist, length, and width) to make them appear as if they belong to a single hand. This consistency ensures clarity and readability.

5. **Simplified Details:**  
   The details of the illustrations have been simplified as much as possible while maintaining clarity, allowing for writing long texts in small sizes without losing legibility.

6. **Letter Thickness:**  
   The appropriate letter thickness for drawing hands has been studied to ensure the typeface remains clear and readable even at small sizes.

### Project Goals:
- Facilitating Quran Learning for the Deaf
- Enhancing Visual Creativity
- Providing an Effective Educational Tool

---

### App Features:
- 🖐 **Real-time Sign Detection:** Detects and tracks Arabic sign language letters live.
- 🎯 **YOLO Model:** Uses a pre-trained YOLOV8 model for accurate detection.
- 📹 **Webcam Integration:** Start and stop your camera to see live predictions.
- 📝 **Annotated Output:** Shows detected signs with bounding boxes and labels.

### How to Use:
1. Check the **Start Camera** box to activate your webcam.
2. Show Arabic sign letters in front of your camera.
3. See live tracking and predictions directly on the screen.
4. Uncheck the box to stop the camera.

This app is ideal for learning, demonstrating, and testing **Arabic Sign Language detection** in real-time.
""")
st.image("a4c424218792587.67a74fe850ce9.png", use_container_width =True)

st.image("8eef4c218792587.67a74fe85037a.png", use_container_width =True)

st.image("383924218792587.67b1be33c0d00.png", use_container_width =True)

st.image("8ed8c9218792587.67a74fe84ebc4.png", use_container_width =True)

st.image("2d60ea218792587.67a74fe84e4fc.png", use_container_width =True)

st.image('8f9011218792587.67ad1df2d1b87 (1).gif', use_container_width=True)

st.image('0acaf8218792587.67a7604bc63cd (1).gif', use_container_width=True)

st.image('dc0952218792587.67a7647e00fa2.png', use_container_width=True)



# Load model
model = YOLO("best.pt")

st.title("**📸 Turn on the Camera**")

start = st.checkbox("Start Camera")

col1, col2 = st.columns(2)

frame_placeholder = col1.empty()

img = Image.open("62dc719041e736da630b18103d9d5dc3.jpg")
img_resized = img.resize((640, 480))
col2.image(img_resized, caption="Arabic Sign Language Letters")

if start:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame")
            break

        results = model.track(frame, persist=True)

        annotated_frame = results[0].plot(labels=False)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(annotated_frame)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("arial.ttf", 28)  

        boxes = results[0].boxes
        names_map = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            label_en = names_map[cls_id]
            label_ar = arabic_map.get(label_en, label_en)

            reshaped_text = arabic_reshaper.reshape(label_ar)
            bidi_text = get_display(reshaped_text)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.text((x1, y1 - 30), bidi_text, font=font, fill=(255, 255, 255))

        annotated_frame = np.array(img_pil)

        annotated_frame = cv2.resize(annotated_frame, (640, 480))

        frame_placeholder.image(annotated_frame, channels="RGB")

        if not st.session_state.get("Start Camera", True):
            break

    cap.release()