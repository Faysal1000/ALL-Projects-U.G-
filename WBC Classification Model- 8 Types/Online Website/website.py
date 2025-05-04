import streamlit as st
import numpy as np
import pickle
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectPercentile, f_classif

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = [ "Basophil",
                "Eosinophil",
                "Erythroblast",
                "Immature granulocytes (IG)",
                "Lymphocyte",
                "Monocyte",
                "Neutrophil",
                "Platelet" ]

def apply_clahe(img):
    """Apply CLAHE after Adaptive Gray World Normalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.02, tileGridSize=(8,8))
    l = clahe.apply(l)
    clahe_img = cv2.merge((l, a, b))
    return cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)

# Load models
@st.cache_resource
def load_models():
    feature_extractor = tf.keras.models.load_model("feature_extractor.keras")

    with open('feature_selector.pkl', 'rb') as file:
        feature_selector = pickle.load(file)
    
    with open('knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    
    return feature_extractor, feature_selector, knn_model

feature_extractor, feature_selector, knn_model = load_models()

# Image preprocessing
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.uint8)
    img = apply_clahe(img)  # Step 1: CLAHE
    img = (img - np.min(img)) / (np.ptp(img) + 1e-8) # step: 2: min-max normalization
    return img  

# Prediction function
def predict_image_class(img, feature_extractor, feature_selector, classifier):
    img_processed = preprocess_image(img)
    features = feature_extractor.predict(tf.expand_dims(img_processed, axis=0), verbose=0)
    selected_features = feature_selector.transform(features)
    prediction_probs = classifier.predict_proba(selected_features)[0]
    return dict(zip(CLASS_NAMES, prediction_probs))

# Streamlit UI
st.title("White Blood Cell Classification")
st.markdown("""
**Upload a Peripheral Blood (PB) image for WBC Classification**  
Supported formats: JPG, JPEG, PNG
""")

uploaded_file = st.file_uploader("Choose a PB image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        predictions = predict_image_class(img_rgb, feature_extractor, feature_selector, knn_model)
        predicted_class = max(predictions, key=predictions.get)
        
        # Create two columns with 3:7 ratio
        col1, col2 = st.columns([3, 7])
        
        with col1:
            st.markdown("<h5 style='margin-bottom:10px;'>Image Preview</h5>", unsafe_allow_html=True)
            st.image(img_rgb, width=230)  
        
        with col2:
            st.markdown("<h5 style='margin-bottom:10px;'>Classification Confidence</h5>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 4)) 
            values = [predictions[cls] for cls in CLASS_NAMES]
            
            # Horizontal bar plot
            bars = sns.barplot(x=values, y=CLASS_NAMES, palette="mako", ax=ax)
            
            # Plot styling
            ax.set_xlabel("Confidence Score", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.tick_params(axis='both', labelsize=12)
            
            # Value annotations
            for i, value in enumerate(values):
                ax.text(value + 0.02, i, f"{value:.2f}",
                        va='center', ha='left',
                        color='white', fontsize=9,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            
            # Clean borders
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)

        # Diagnosis result below
        st.markdown(f"""
        <div style='background-color:#e8f4f8; padding:20px; border-radius:10px; margin-top:20px;'>
            <h3 style='color:#2c3e50; margin-bottom:15px;'> Clinical Diagnosis</h3>
            <p style='font-size:20px; color:#2980b9; margin-bottom:5px;'>
            🩸 <strong>Primary Classification:</strong> {predicted_class}
            </p>
            <p style='font-size:16px; color:#2c3e50;'>
            📈 Confidence Level: {predictions[predicted_class]*100:.1f}%<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")