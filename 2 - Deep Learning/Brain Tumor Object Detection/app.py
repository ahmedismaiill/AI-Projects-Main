import streamlit as st
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit
st.cache_data.clear()
st.set_page_config(page_title="Brain Tumor Object Detection",page_icon='🧠')

# Set the title of the Streamlit app
st.markdown(
    "<h1 style='font-size:2.5em;'>"
    "<span style='color: red;'>Brain Tumor</span> Detection App 🧠"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown("**Upload an MRI and the app detects & localizes tumors with YOLO-style bounding boxes, classifying into Glioma, Meningioma, No Tumor, or Pituitary.Displays prediction — Model accuracy: **96%** 🔎🚀.**")
st.image('Brain-Tumor.gif', use_container_width=True)
st.divider()


# Process the uploaded images
st.title("📱 Upload MRI of Brain")
MRI_Brain = st.file_uploader("**MRI of Brain**", type=["jpg", "jpeg", "png"])

def process_image(MRI_Brain):
    if MRI_Brain is not None:
        # Read the image file
        image = np.array(bytearray(MRI_Brain.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        st.image(image, channels="RGP", caption="MRI Image", use_container_width=True)
        # Display the processed image
        st.divider()
        st.subheader("Processed MRI Image:")

        col1 , col2 = st.columns(2)

        if image is not None:
            # Perform inference using the YOLO model
            results = model(image, imgsz=640, conf=0.25, device='cuda')
            # Process the results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls)]
                    color = (0, 255, 0) if label == "No Tumor" else (255, 0, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the original image in the first column
            with col1:
                st.image(image, channels="RGP", caption="Processed MRI Image", use_container_width=True)

            with col2:
                # Display only the bounding boxes and labels cut from the original image    
                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls)]
                    st.image(image[y1:y2, x1:x2], caption=f"{label} {score:.2f}", use_container_width=True)
            # Show Predictions
            st.subheader("Predictions:")
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                st.write(f"## 🔎 {label} {score:.2f} %")
                col2.metric(f"**{label}**", f"{score:.2f} %")
            # Show success message
            st.success("Tumor detection completed successfully!")
        else:
            st.error("Error reading the image. Please upload a valid image file.")
            


process_image(MRI_Brain)
