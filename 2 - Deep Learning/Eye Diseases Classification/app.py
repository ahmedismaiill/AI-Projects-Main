import cv2 
import keras 
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the pre-trained model
model = load_model('model_EfficientNetB7.h5')
Class_Names_Dict = {'Glaucoma': 0, 'Normal': 1, 'Diabetic Retinopathy': 2, 'Cataract': 3}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit
st.cache_data.clear()
st.set_page_config(page_title="Heart Failure Prediction",page_icon='👁️')

st.title("Eye Diseases Classification 👁️")
st.markdown("**This project uses a deep learning model based on EfficientNetB7 to classify retinal images into four categories: Normal, Diabetic Retinopathy, Cataract, and Glaucoma The model achieves an overall accuracy of 95%, demonstrating strong performance in detecting various eye conditions.🚨**")
st.image('diabetic-eye-issues-5-ways-diabetes-impacts-vision.jpg')
st.divider()

st.sidebar.markdown("## 🩺 Eye Disease Descriptions")

st.sidebar.markdown("""
### 👁️ Glaucoma  
Damage to the optic nerve, often caused by high intraocular pressure.  
It can lead to gradual, irreversible vision loss if untreated.

---

### 👁️ Diabetic Retinopathy  
Caused by diabetes damaging the retina’s blood vessels.  
May lead to blurred vision and blindness without early treatment.

---

### 👁️ Cataract  
Clouding of the eye's lens, usually due to aging.  
It causes blurry vision and glare, treatable with surgery.

---

### 👁️ Normal  
Healthy eye with no signs of disease or retinal abnormalities.  
Vision remains clear and unaffected.
""")


st.title("Upload Left Eye")
Left_Eye = st.file_uploader("**Left Eye**", type=["jpg", "jpeg", "png"])

st.title("Upload Right Eye")
Right_Eye = st.file_uploader("**Right Eye**", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

with col1:
    if Left_Eye is not None:
        st.image(Left_Eye, caption="Left Eye", use_container_width=True)
        

with col2:
    if Right_Eye is not None:
        st.image(Right_Eye, caption="Right Eye", use_container_width=True)

if Right_Eye is not None and Left_Eye is not None:
  def predict_image(image):
      image = Image.open(image).convert('RGB')    
      image = image.resize((224, 224))                    
      image_np = np.array(image, dtype='float32')   
      image_np = preprocess_input(image_np)            
      image_batch = np.expand_dims(image_np, axis=0)     

      # Predict
      probs = model.predict(image_batch)[0]
      predicted_class_index = np.argmax(probs)
      return predicted_class_index, probs

  predicted_class_index_left, prop_left = predict_image(Left_Eye)
  predicted_class_index_right, prop_right = predict_image(Right_Eye)

  class_names = list(Class_Names_Dict.keys())

  st.subheader("Prediction Results")

  col1, col2 = st.columns(2)

  with col1:
        st.markdown("### Left Eye Prediction")
        st.write("Predicted Class:", class_names[predicted_class_index_left])
        for i, prob in enumerate(prop_left):
            st.write(f"{class_names[i]}: {prob:.2%}")
        col1.metric(f"{class_names[predicted_class_index_left]}", f"{prop_left[predicted_class_index_left]*100:.2f} %")

  with col2:
        st.markdown("### Right Eye Prediction")
        st.write("Predicted Class:", class_names[predicted_class_index_right])
        for i, prob in enumerate(prop_right):
            st.write(f"{class_names[i]}: {prob:.2%}")
        col2.metric(f"{class_names[predicted_class_index_right]}", f"{prop_right[predicted_class_index_right]*100:.2f} %")