import streamlit as st
loader_placeholder = st.empty()
loader_placeholder.markdown("""
<div style="
    display:flex;
    justify-content:center;
    align-items:center;
    height:50vh;
    font-size:40px;
    font-weight:bold;
    color:#00b4d8;
    animation: flash 1s infinite;
">
Loading necessary libraries...
</div>
<style>
@keyframes flash {
  0% { opacity: 0.2; }
  50% { opacity: 1; }
  100% { opacity: 0.2; }
}
</style>
""", unsafe_allow_html=True)
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Layer, Dense, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Add, Activation, Reshape, Multiply, Concatenate
)
from tensorflow.keras import initializers

import matplotlib.pyplot as plt

# Keras serialization
from tensorflow.keras.utils import register_keras_serializable

loader_placeholder.empty()

# ==============================================================
# 0. Extra Config for Loading The Model
# ==============================================================
@register_keras_serializable(package="Custom", name="Res2Split")
class Res2Split(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        channels = tf.shape(x)[-1]
        chunk = channels // self.scale
        splits = []
        for i in range(self.scale):
            splits.append(x[:, :, :, i*chunk:(i+1)*chunk])
        return splits

# Channel Attention Block
@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelAttention")
class ChannelAttention(Layer):
    def __init__(self, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channel = input_shape[-1]  # Number of channels in the input tensor
        self.shared_dense_one = Dense(channel // self.reduction, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    def call(self, inputs):
        # Global average and max pooling
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalMaxPooling2D()(inputs)

        # Apply shared dense layers for both average and max pooled features
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        # Combine both attention signals and apply sigmoid activation
        attention = Add()([avg_pool, max_pool])
        attention = Activation('sigmoid')(attention)

        # Reshape the attention to match the input dimensions and apply multiplication
        attention = Reshape((1, 1, -1))(attention)
        return Multiply()([inputs, attention])

# Spatial Attention Block
@tf.keras.utils.register_keras_serializable(package="Custom", name="SpatialAttention")
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        # 2D convolution for spatial attention with a kernel size of 7x7 and sigmoid activation
        self.conv2d = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        # Apply global average and max pooling along the channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate both pooled features
        concat = Concatenate(axis=-1)([avg_pool, max_pool])

        # Apply convolution to compute spatial attention map
        attention = self.conv2d(concat)

        # Multiply the attention map with the input to highlight important spatial features
        return Multiply()([inputs, attention])

# Full CBAM Block
def cbam_block(inputs, reduction=16):
    # Apply Channel Attention followed by Spatial Attention
    x = ChannelAttention(reduction)(inputs)
    x = SpatialAttention()(x)
    return x

@register_keras_serializable(package="Custom", name="KANLayer")
class KANLayer(Layer):
    def __init__(self, units, grid=5, spline_order=3, dropout=0.0, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = int(units)
        self.grid = int(grid)
        self.spline_order = int(spline_order)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"KANLayer expects 2D input (batch, features). Got shape: {input_shape}")

        input_dim = int(input_shape[-1])

        # Linear weights (like Dense)
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=initializers.HeNormal(),
            trainable=True,
            name="linear_weight"
        )

        # Spline weights: (grid, input_dim, units)
        self.spline_w = self.add_weight(
            shape=(self.grid, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="spline_weight"
        )

        # Bias
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

        super(KANLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)

        # --- Linear part ---
        linear_out = tf.linalg.matmul(x, self.w)

        # --- Spline / soft-binning part ---
        x_exp = tf.expand_dims(x, axis=1)  # (B, 1, D)

        # Fixed 0-1 grid (assumes input normalized to 0-1)
        grid_centers = tf.linspace(0.0, 1.0, self.grid)
        grid_centers = tf.reshape(grid_centers, (1, self.grid, 1))  # (1, grid, 1)
        grid_centers = tf.cast(grid_centers, tf.float32)

        # Soft RBF-like basis: (B, grid, D)
        spline_basis = tf.exp(-tf.square(x_exp - grid_centers))

        # einsum: "bgd,gdu->bu" -> (B, units)
        spline_out = tf.einsum("bgd,gdu->bu", spline_basis, self.spline_w)

        out = linear_out + spline_out + self.b

        if self.dropout and training:
            out = tf.nn.dropout(out, rate=self.dropout)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        cfg = super(KANLayer, self).get_config()
        cfg.update({
            "units": self.units,
            "grid": self.grid,
            "spline_order": self.spline_order,
            "dropout": self.dropout,
        })
        return cfg

@tf.keras.utils.register_keras_serializable(package="Custom", name="F1Score")
class F1Score(tf.keras.metrics.Metric):


    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()  # Precision metric
        self.recall = tf.keras.metrics.Recall()  # Recall metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()  # Get precision value
        r = self.recall.result()  # Get recall value
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())  # Calculate F1 score

    def reset_states(self):

        self.precision.reset_states()  # Reset precision state
        self.recall.reset_states()  # Reset recall state


# ==============================================================
# 2. CONSTANTS
# ==============================================================

IMG_SIZE = 224
CLASS_NAMES = ['Benign', 'Early Pre-B', 'Pre-B', 'Pro-B']
TARGET_CONV_LAYER = 'activation_25'  # update if needed

# ==============================================================
# 3. SEGMENTATION HELPERS
# ==============================================================

def RGB2LAB(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image_lab)
    return l, a, b

def get_mask(image):
    l, a, b = RGB2LAB(image)
    a_blur = cv2.GaussianBlur(a, (19, 19), 0)
    _, thresh = cv2.threshold(a_blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

# ==============================================================
# 4. PREPROCESSING
# ==============================================================

def preprocess_image(image):
    """
    Simple Min-Max Normalization for RGB image.
    Returns float32 image scaled to [0, 1].
    """
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    # avoid division by zero
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.float32)

    # min-max scale to [0, 1]
    image = (image - min_val) / (max_val - min_val)

    return image

# ==============================================================
# 5. GRAD-CAM++ EXPLAINER
# ==============================================================

class GradCAMPlusPlus:
    """Grad-CAM++ for better attribution maps"""
    
    def __init__(self, model, target_layer, labels):
        self.model = model
        self.target_layer_name = target_layer
        self.labels = labels
        
        try:
            self.target_layer = model.get_layer(self.target_layer_name)
        except:
            print(f"❌ Layer '{self.target_layer_name}' not found!")
            print(f"Available Conv layers: {[l.name for l in model.layers if 'Conv' in l.__class__.__name__]}")
            raise
        
        print(f"✓ Grad-CAM++ initialized with layer: {self.target_layer_name}")

    def compute_gradcam_plus_plus(self, image, class_idx):
        """Compute Grad-CAM++ activation map"""
        img_tensor = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), 0)
        
        intermediate_model = Model(
            inputs=self.model.input,
            outputs=[self.target_layer.output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            layer_output, predictions = intermediate_model(img_tensor)
            tape.watch(layer_output)
            loss = predictions[0, class_idx]
        
        gradients = tape.gradient(loss, layer_output)
        
        if gradients is None:
            print("⚠️ Gradients None - using fallback")
            return self._compute_fallback_cam(image, class_idx, layer_output.numpy())
        
        layer_output_np = layer_output.numpy()[0]
        gradients_np = gradients.numpy()[0]
        
        second_deriv = np.power(gradients_np, 2)
        third_deriv = second_deriv * gradients_np
        
        alpha_denom = 2 * second_deriv + np.sum(third_deriv * layer_output_np, axis=(0, 1), keepdims=True)
        alpha_denom = np.where(alpha_denom != 0, alpha_denom, 1e-8)
        
        alpha = second_deriv / alpha_denom
        alpha = np.maximum(alpha, 0)
        
        weights = np.sum(alpha * np.maximum(gradients_np, 0), axis=(0, 1))
        weights = weights / (np.sum(weights) + 1e-8)
        
        cam = np.zeros(layer_output_np.shape[:2])
        for i, w in enumerate(weights):
            cam += w * layer_output_np[:, :, i]
        
        cam = np.maximum(cam, 0)
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def _compute_fallback_cam(self, image, class_idx, layer_output):
        """Fallback CAM computation"""
        layer_output = layer_output[0]
        weights = np.mean(np.abs(layer_output), axis=(0, 1))
        weights = weights / (np.sum(weights) + 1e-8)
        
        cam = np.zeros(layer_output.shape[:2])
        for i, w in enumerate(weights):
            cam += w * layer_output[:, :, i]
        
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def predict_and_explain(self, image):
        """Complete pipeline"""
        img_tensor = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), 0)
        
        preds = self.model.predict(img_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        
        gradcam = self.compute_gradcam_plus_plus(image, pred_idx)
        
        gradcam_resized = tf.image.resize(
            tf.expand_dims(gradcam, -1), image.shape[:2]
        ).numpy()[..., 0]
        
        return {
            'prediction': self.labels[pred_idx],
            'confidence': confidence,
            'gradcam': gradcam_resized,
            'preds': preds
        }

# helper: create overlay image (no matplotlib figure, good for Streamlit)
def create_gradcam_overlay(image, gradcam):
    # image assumed float [0,1] or uint8
    img_normalized = image / 255.0 if image.max() > 1 else image
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(gradcam)[..., :3]
    alpha_blend = gradcam[..., np.newaxis]
    overlay = (1 - alpha_blend) * img_normalized + alpha_blend * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8

# ==============================================================
# 6. LOAD MODEL & EXPLAINER (CACHED)
# ==============================================================

@st.cache_resource
def load_trained_model():
    model = load_model('Proposed_Model.keras')
    return model

@st.cache_resource
def get_explainer():
    model = load_trained_model()
    explainer = GradCAMPlusPlus(
        model=model,
        target_layer=TARGET_CONV_LAYER,
        labels=CLASS_NAMES
    )
    return explainer


# ==============================================================
# UI / APP (Improved Design)
# ==============================================================

st.set_page_config(
    page_title="Blood Smear Classifier (Prototype)",
    page_icon="🩸",
    layout="wide",
)

# Small CSS to improve look (cards, fonts, spacing)
st.markdown(
    """
    <style>
    .title {font-size:38px; font-weight:700; margin:0}
    .subtitle {color: #6b7280; margin-top:0}
    .card {background: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.08);}
    .small {font-size:12px; color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True,
)

# Top header area
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<div class="title">🩸 Blood Smear Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Research prototype — visualize model attention and class confidences</div>', unsafe_allow_html=True)

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
logo_path = "blood_icon.png"
logo_base64 = get_base64_image(logo_path)

with header_col2:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center;">
            <img src="data:image/png;base64,{logo_base64}" style="
                max-width:40px;
                width:5vw;
                min-width:10px;
            ">
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("---")

# Sidebar: controls & model info
with st.sidebar:
    st.markdown("### Model & Settings")
    st.info("Model: Proposed_Model.keras")
    st.checkbox("Show segmentation mask on diagnostics", value=False, key="show_mask")
    st.write("---")
    st.markdown("### Tips")
    st.markdown("- Upload a clear single-cell / smear image.\n- The Grad-CAM overlay highlights regions the model used for prediction.")
    st.write("---")
    st.markdown("### About")
    st.caption("Experimental — not for clinical use")

# File uploader
uploaded_file = st.file_uploader(
    "Upload blood smear image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Please upload a microscopic blood smear image to begin.")
else:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("Could not read the uploaded image. Please try a different file.")
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Show diagnostics and processing controls in an expander
        with st.expander("Preview & preprocessing options", expanded=False):
            colA, colB = st.columns([2, 1])
            with colA:
                st.image(rgb, caption="Original (preview)", use_container_width=True)
            with colB:
                st.markdown("**Resize**: 224×224 (model input)")
                st.markdown("**Segmentation**: color-threshold based (LAB a-channel)")

        # Do segmentation, preprocess, predict inside a spinner
        with st.spinner("Running model & computing Grad-CAM++..."):
            mask = get_mask(rgb)
            segmented = apply_mask(rgb, mask)
            resized = cv2.resize(segmented, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            preprocessed = preprocess_image(resized)  # float [0,1]

            explainer = get_explainer()
            result = explainer.predict_and_explain(preprocessed)

        preds = result["preds"]
        gradcam = result["gradcam"]
        pred_label = result["prediction"]
        pred_conf = result["confidence"]

        overlay = create_gradcam_overlay(preprocessed, gradcam)

        # Layout: left — images, right — summary + download
        images_col, info_col = st.columns([2, 1], gap="large")

        with images_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Image views**")
            img1, img2, img3 = st.columns(3)

            with img1:
                st.markdown("**Original (resized)**")
                original_resized_display = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                st.image(original_resized_display, use_container_width =True)

            with img2:
                st.markdown("**Segmented**")
                st.image(resized, use_container_width =True)

            with img3:
                st.markdown("**Model Attention**")
                st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Extra: show mask if user asked for it
            if st.session_state.get("show_mask", False):
                st.markdown("**Segmentation mask**")
                st.image(mask, use_container_width=True)

            if not st.session_state.get("show_mask", False):
                st.write("---")
                st.markdown("### Class confidences")

                col1, col2, col3, col4 = st.columns(4)

                cols = [col1, col2, col3, col4]

                for idx, class_name in enumerate(CLASS_NAMES):
                    score = float(preds[idx])

                    with cols[idx]:
                        st.metric(label=class_name, value=f"{score*100:.1f}%")
                        bar = st.progress(0)
                        bar.progress(int(score * 100))


        with info_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Prediction")
            st.markdown(f"**{pred_label}**  ")
            st.markdown(f"Confidence: **{pred_conf:.1%}**")
            
            if st.session_state.get("show_mask", False):
                st.write("---") 
                st.markdown("### Class confidences") 
                for i, class_name in enumerate(CLASS_NAMES): 
                    score = float(preds[i]) 
                    st.metric(label=class_name, value=f"{score*100:.1f}%") 
                    bar = st.progress(0) 
                    bar.progress(int(score * 100))


            st.write("---")
            # Image download
            st.markdown("### Export")
            is_success, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if is_success:
                b = buffer.tobytes()
                st.download_button("Download overlay (PNG)", data=b, file_name="gradcam_overlay.png", mime="image/png")

            st.markdown('</div>', unsafe_allow_html=True)

        st.caption("This tool is experimental and should only be used for research, education, or model debugging — not for patient care.")

# -------------------------
# End of UI file
# -------------------------