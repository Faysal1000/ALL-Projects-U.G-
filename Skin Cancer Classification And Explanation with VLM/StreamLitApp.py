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
from st_click_detector import click_detector
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
import base64
import os
import io
import traceback
from tensorflow.keras.layers import (
    Layer, Conv2D, Dense,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    Reshape, Multiply, Add, Activation, Concatenate
)
from pathlib import Path

loader_placeholder.empty()
#--------------------------------------------------------------------------------------------------
# unnecessary for this app, but needed for CNN model to load, so its necessary actually 
#--------------------------------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Custom", name="F1Score")
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()  
        self.recall = tf.keras.metrics.Recall() 

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()  
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())  

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states() 

@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelAttention")
class ChannelAttention(Layer):
    def __init__(self, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channel = input_shape[-1] 
        self.shared_dense_one = Dense(channel // self.reduction, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalMaxPooling2D()(inputs)

        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        attention = Add()([avg_pool, max_pool])
        attention = Activation('sigmoid')(attention)

        attention = Reshape((1, 1, -1))(attention)
        return Multiply()([inputs, attention])

@tf.keras.utils.register_keras_serializable(package="Custom", name="SpatialAttention")
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv2d = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv2d(concat)
        return Multiply()([inputs, attention])
    
def cbam_block(inputs, reduction=16):
    x = ChannelAttention(reduction)(inputs)
    x = SpatialAttention()(x)
    return x
#----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# -------------------------
# Helpers & small utilities
# -------------------------
def bytes_from_path(path):
    with open(path, "rb") as f:
        return f.read()

def image_to_data_uri(path: str, max_width=224, jpeg_quality=70):
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    # resize maintaining aspect ratio
    if img.width > max_width:
        new_h = int(max_width * img.height / img.width)
        img = img.resize((max_width, new_h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b = buf.getvalue()
    data64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{data64}"


labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
full_names = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

def preprocess_image(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image_clahe = image_clahe.astype(np.float32)
    image_clahe = (image_clahe - np.min(image_clahe)) / (np.ptp(image_clahe) + 1e-8)
    return image_clahe

@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path="Proposed CBAM-Xception-DermNet.keras"):
    if 'cnn_model' in st.session_state:
        return st.session_state.cnn_model
    try:
        model = load_model(model_path)
        st.session_state.cnn_model = model
        return model
    except Exception as e:
        st.error(f"Failed to load CNN model from '{model_path}': {e}")
        st.exception(traceback.format_exc())
        raise

@st.cache_resource(show_spinner=False)
def load_vlm_model():
    if st.session_state.get("vlm_loaded", False):
        return {
            "model": st.session_state.vlm_model,
            "processor": st.session_state.processor,
            "device": st.session_state.device,
            "dtype": st.session_state.dtype
        }

    USE_4BIT = True              
    HF_MODEL_ID = "google/medgemma-4b-it"  # Hugging Face repo ID
    LORA_OUTPUT_DIR = "./medgemma_lora_adapter" #local lora saved dir
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN") #NOTE: hiding mandatory (reminder)

    # Determine dtype
    capability = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if torch.cuda.is_available() and capability >= 8 else torch.float32

    # 4-bit quantization config
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    # Load processor from LoRA adapter folder (it contains tokenizer, etc.)
    try:
        processor = AutoProcessor.from_pretrained(
            LORA_OUTPUT_DIR,
            trust_remote_code=True
        )
        processor.tokenizer.padding_side = "right"
    except Exception as e:
        st.error(f"Failed to load processor from '{LORA_OUTPUT_DIR}': {e}")
        st.exception(traceback.format_exc())
        raise

    # Load base model from Hugging Face hub
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            HF_MODEL_ID,
            quantization_config=bnb_config if USE_4BIT else None,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token  
        )
    except Exception as e:
        st.error(f"Failed to load base model from Hugging Face hub: {e}")
        st.exception(traceback.format_exc())
        raise

    # Attach LoRA adapter
    try:
        model = PeftModel.from_pretrained(
            base_model,
            LORA_OUTPUT_DIR,
            device_map="auto"
        )
    except Exception as e:
        st.error(f"Failed to attach LoRA adapter: {e}")
        st.exception(traceback.format_exc())
        raise
    model.eval()
    try:
        model.to(DEVICE)
    except Exception:
        # ignore if model already on correct device
        pass
    # Cache into session_state
    st.session_state.vlm_model = model
    st.session_state.processor = processor
    st.session_state.device = DEVICE
    st.session_state.dtype = dtype
    st.session_state.vlm_loaded = True

    return {"model": model, "processor": processor, "device": DEVICE, "dtype": dtype}


def generate_vlm_response(processor, vlm_model, device, gradcam_image: Image.Image, pred_label,
                          max_new_tokens=128):
    try:
        prompt_template = (
            "You are an AI assistant specialized in model interpretability. "
            "I am providing:\n- CNN model Grad-CAM++ heatmap image\n- Model predicted class: {predicted_class}\n\n"
            "Based on the Grad-CAM++ heatmap, write a clear and concise 20–30 word explanation "
            "of which features the model focused on and why. Output only the explanation (no headings)."
        )
        user_prompt = prompt_template.format(predicted_class=pred_label)

        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ],
            }
        ]
        formatted_prompt = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=formatted_prompt, images=gradcam_image, return_tensors="pt", padding=True)

        try:
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

        if hasattr(inputs, "pixel_values") or ("pixel_values" in inputs):
            try:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vlm_model.dtype)
            except Exception:
                try:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
                except Exception:
                    pass

        with torch.inference_mode():
            output_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        # Some generate wrappers return object with .sequences
        if hasattr(output_ids, "sequences"):
            seqs = output_ids.sequences
        else:
            seqs = output_ids

        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(seqs[0, input_len:], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        st.error(f"VLM generation failed: {e}")
        st.exception(traceback.format_exc())
        return None

def classify_and_gradcam(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocessed = preprocess_image(np.array(pil_img))
    input_tensor = np.expand_dims(preprocessed, axis=0)
    with st.spinner("Loading Classifier Model..."):
        cnn = load_cnn_model("Proposed CBAM-Xception-DermNet.keras")
    with st.spinner("Classifying..."):
        preds = cnn.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = labels[pred_idx]
        conf = float(preds[pred_idx])
    with st.spinner("Generating Attention Map..."):
        target_layer = "block14_sepconv2"
        score = CategoricalScore([pred_idx])
        gradcam_vis = GradcamPlusPlus(cnn, model_modifier=ReplaceToLinear(), clone=True)
        cam = gradcam_vis(score, input_tensor, penultimate_layer=target_layer)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = plt.cm.jet(cam)[..., :3]
        overlay = 0.25 * heatmap + 0.75 * preprocessed
        overlay = np.uint8(255 * np.clip(overlay, 0, 1))
        overlay_pil = Image.fromarray(overlay)

    return pred_label, conf, overlay_pil

# -------------------------
# Main display config & styling 
# -------------------------
st.set_page_config(page_title="Skin Cancer Classifier", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f5f7fb 0%, #ffffff 100%); }
.card { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 8px 22px rgba(14,30,37,0.06); }
.header-title { font-size:34px; font-weight:700; margin-bottom:4px; }
.header-sub { color:#6b7280; margin-bottom:6px; }
.small { font-size:13px; color:#6b7280; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Important Notice")
    st.markdown("""
    - This app is a prototype, not for clinical use.
    - Do not rely on classifications or explanations for medical decisions.
    - This apps model is fine tuned on only one small dataset.
    - It might not capture your original disease.
    - Always consult a qualified healthcare professional.
    - Results may not be accurate; use at your own risk.
    - Again, this is just a prototype!
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Clear Models Cache"):
        for k in ["cnn_model", "vlm_model", "processor", "device", "dtype", "vlm_loaded"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Model cache cleared. Models will reload on next use.")


st.markdown("<div class='header-title'>Skin Cancer Image Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>CNN inference • Model Attention (Grad-CAM++) visualizations • VLM explanations</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg","jpeg","png"], key="uploaded_file" )

# --- Handle automatic reset if file is cleared ---
#if uploaded_file is None and "selected_image" in st.session_state:
#    # Only clear if user manually removed an uploaded file
#    if not st.session_state.get("example_selected", False):
#        for key in ["selected_image", "vlm_response"]:
#            st.session_state.pop(key, None)
#        st.rerun()

if uploaded_file is not None:
    st.session_state.selected_image = uploaded_file.read()
    st.session_state.example_selected = False  
    st.session_state["vlm_response"] = None  

if uploaded_file is None and not st.session_state.get("example_selected", False):
    keys_to_clear = ["vlm_response", "pred_label", "conf", "overlay_pil", "last_image_bytes", "selected_image"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

# Main layout: image area and visualization
original_image_col, attention_column = st.columns([2,2])

with original_image_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Selected Image")
    if 'selected_image' in st.session_state:
        pil_img = Image.open(io.BytesIO(st.session_state.selected_image)).convert("RGB")
        st.image(pil_img, width=360, caption="Selected image", output_format="auto")
    else:
        st.info("No image selected. Upload or click an example below.")
    st.markdown("</div>", unsafe_allow_html=True)

# full column
if 'selected_image' in st.session_state:
    img_bytes = st.session_state.selected_image
    if st.session_state.get("last_image_bytes") != img_bytes:
        pred_label, conf, overlay_pil = classify_and_gradcam(img_bytes)
        st.session_state["pred_label"] = pred_label
        st.session_state["conf"] = conf
        st.session_state["overlay_pil"] = overlay_pil
        st.session_state["last_image_bytes"] = img_bytes
        st.session_state["vlm_response"] = None
        
with attention_column:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Attention Visualization")
    if 'selected_image' in st.session_state:
        st.image(st.session_state["overlay_pil"], caption="Model Attention Overlay", width=360, output_format="auto")
    else:
        st.info("Model Attention will appear here after selecting an image and running classification.")
    st.markdown("</div>", unsafe_allow_html=True)


# Metrics placeholder
c1, c2 = st.columns([3,1])
if st.session_state.get("selected_image") and st.session_state.get("pred_label"):
    c1.metric("Predicted", full_names[st.session_state["pred_label"]])
    c2.metric("Confidence", f"{st.session_state['conf']:.2f}")
else:
    c1.metric("Predicted", "—")
    c2.metric("Confidence", "—")

if st.button("Generate VLM Explanation"):
    if 'selected_image' in st.session_state:
        if not st.session_state.get("vlm_response", False):
            try:
                with st.spinner("Loading VLM Model. First time load will take time. Please be patient..."):
                    try:
                        vlm_info = load_vlm_model()
                    except Exception as e:
                        st.error("VLM load failed. See logs above.")
                        vlm_info = None

                if vlm_info is not None:
                    try:
                        img_for_vlm = st.session_state["overlay_pil"].convert("RGB").resize((224, 224), Image.BILINEAR)
                    except Exception:
                        st.warning("Overlay image not available for VLM input; using original image.")
                        img_for_vlm = st.session_state["selected_image"].convert("RGB").resize((224, 224), Image.BILINEAR)

                    with st.spinner("Generating explanation... (NOTE: This may take a moment on the free version)"):
                        response = generate_vlm_response(
                            vlm_info["processor"],
                            vlm_info["model"],
                            vlm_info["device"],
                            img_for_vlm,
                            full_names[st.session_state["pred_label"]],
                            max_new_tokens=128
                        )

                    #response = "Debugging VLM response."
                    if response is None:
                        st.error("VLM did not return a response.")
                    else:
                        st.session_state["vlm_response"] = response

            except Exception as e:
                st.error(f"Error in VLM generation flow: {e}")
                st.exception(traceback.format_exc())
    else:
        st.warning("Upload an image first or use the example images provided below!")


if st.session_state.get("vlm_response", False):
    st.subheader("Generated Explanation")
    if st.session_state.get("vlm_response"):
        st.info(st.session_state["vlm_response"])
    else:
        st.info("VLM explanation will appear here after selecting an image and running classification.")

example_paths = [
    "images/ISIC_0025314.jpg",
    "images/ISIC_0025586.jpg",
    "images/ISIC_0025680.jpg",
    "images/ISIC_0026163.jpg"
]

# Container div for toggle + gallery
st.markdown("""
<div style='background-color:#f9fafb; padding:15px; border-radius:12px; margin-bottom:20px;'>
""", unsafe_allow_html=True)

toggle = st.toggle("Show Example Images", value=False)

if toggle:
    # Toggle ON → show gallery
    st.markdown("<div class='header-sub'>Click on any image to analyze it instantly</div>", unsafe_allow_html=True)
    html = """
    <style>
    .example-img {
        border-radius:10px;
        width:100%;
        display:block;
        box-shadow: 0 4px 12px rgba(14,30,37,0.06);
        transition: transform .12s ease, box-shadow .12s ease;
        cursor: pointer;
    }
    .example-img:hover {
        transform: scale(1.03);
        box-shadow: 0 14px 30px rgba(14,30,37,0.10);
    }
    .gallery-row { display:flex; gap:20px; }
    .gallery-item { flex:1; }
    </style>
    <div class="gallery-row">
    """

    for i, path in enumerate(example_paths):
        src = image_to_data_uri(path, max_width=480, jpeg_quality=70)
        if src is None:
            placeholder_svg = """
            <svg xmlns='http://www.w3.org/2000/svg' width='400' height='300'>
                <rect width='100%' height='100%' fill='#f3f4f6'/>
                <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' 
                    fill='#9ca3af' font-size='20'>missing</text>
            </svg>
            """
            src = "data:image/svg+xml;base64," + base64.b64encode(placeholder_svg.encode()).decode()

        html += f"""
        <a href='#' id='img_{i}' class='gallery-item'>
            <img src='{src}' class='example-img' />
        </a>
        """

    html += "</div>"

    if "example_click_key" not in st.session_state:
        st.session_state.example_click_key = 0

    clicked = click_detector(html, key=f"clicking_examples_{st.session_state.example_click_key}")

    if clicked:
        if uploaded_file is not None:
            st.warning("Please remove the uploaded file by clickng cross in the uploaded file name")
        else:    
            idx = int(clicked.split("_")[1])
            selected_path = example_paths[idx]
            img_bytes = open(selected_path, "rb").read()
            if st.session_state.get("last_image_bytes") != img_bytes:
                st.session_state.selected_image = img_bytes
                st.session_state.example_selected = True
                st.session_state["vlm_response"] = None
                st.session_state.example_click_key += 1
                try:
                    st.toast(f"✅ Selected image: {selected_path}", icon="📸")
                except Exception:
                    st.success(f"Selected image: {selected_path}")
                st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:12px; color:#6b7280; font-size:13px;'>
© 2025 Faysal Ahmmed, Ajmy Alaly, Samanta Mehnaj, Asef Rahman, F.M. Mridha. All rights reserved.
</div>
""", unsafe_allow_html=True)
