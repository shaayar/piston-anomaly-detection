import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model/keras_model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.title("Anomaly Detection with Teachable Machine")
st.write("Upload or capture an image to detect if it's normal or anomalous.")

# Add option for user to choose input method
input_method = st.radio(
    "Select image input method:",
    ("Upload Image", "Capture Image")
)

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
        except Exception:
            st.error("Invalid image file.")
elif input_method == "Capture Image":
    captured_img = st.camera_input("Take a picture")
    if captured_img is not None:
        try:
            img = Image.open(captured_img).convert("RGB")
            st.image(img, caption="Captured Image", use_column_width=True)
        except Exception:
            st.error("Invalid image capture.")

if img is not None and model is not None:
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)
        class_names = ["Normal", "Anomaly"]  # Consider loading from file if possible

        pred_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        st.write(f"### Prediction: **{class_names[pred_idx]}**")
        st.write(f"Confidence: {confidence:.2f}%")
elif model is None:
    st.warning("Model not loaded. Please check the model path.")