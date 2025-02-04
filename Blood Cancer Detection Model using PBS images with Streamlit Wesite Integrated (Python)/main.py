import streamlit as st
import numpy as np
import pickle
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply

with open("stacking_model.pkl", "rb") as f:
    stacking_model = pickle.load(f)

IMG_SIZE = (224, 224)
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

CLASS_NAMES = [
    "Benign or non-cancerous",
    "Early-stage Acute Lymphoblastic Leukemia, the disease is in its initial phases",
    "Precursor phase of Acute Lymphoblastic Leukemia, before the leukemia becomes fully developed",
    "Progenitor phase, a stage where the cancerous cells are in a progenitor (early developmental) stage"
]

def get_feature_extractor():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    gap = GlobalAveragePooling2D()(base_model.output)
    se = Dense(base_model.output_shape[-1] // 16, activation="relu")(gap)
    se = Dense(base_model.output_shape[-1], activation="sigmoid")(se)
    se = Reshape((1, 1, -1))(se)
    x = Multiply()([base_model.output, se])
    x = GlobalAveragePooling2D()(x)  # Final feature extraction
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    return feature_extractor

def preprocess_image(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def predict_image_class(img):
    img_resized = preprocess_image(img)  
    features = get_feature_extractor().predict(img_resized, verbose=0)  
    prediction = stacking_model.predict(features)  
    predicted_class = CLASS_NAMES[prediction[0]] 
    return predicted_class

st.title("Detection of Acute Lymphoblastic Leukemia")
st.write("### Upload a PBS image to Classify ALL.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    predicted_class = predict_image_class(img_rgb)
    st.success(f"### 🎯 Prediction: **{predicted_class}**")