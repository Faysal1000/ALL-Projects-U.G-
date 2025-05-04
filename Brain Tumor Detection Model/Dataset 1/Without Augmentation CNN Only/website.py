import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import warnings
warnings.filterwarnings("ignore")

IMG_SIZE = (299,299)
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

classifier = tf.keras.models.load_model("brain_tumor_xception_classifier.keras")


def preprocess_image(image):
    img_resized = cv2.resize(image, IMG_SIZE)
    img = img_resized.astype(np.float32) / 255.0  
    return img

def predict_image_class(img, classifier):
    img_processed = preprocess_image(img)
    prediction = classifier.predict(np.expand_dims(img_processed, axis=0), verbose=1)
    predicted_class = CLASS_NAMES[np.argmax(prediction)] 
    return predicted_class

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

st.title("Detection of Brain Tumor")
st.write("### Upload an M.R.I Photo of Rrain")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    predicted_class = predict_image_class(img_rgb, classifier)
    st.success(f"### 🎯 Prediction: **{predicted_class}**")