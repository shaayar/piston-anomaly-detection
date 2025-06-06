import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import pandas as pd

# Cache the model loading function
@st.cache_resource
def load_model():
    try:
        # Handle the DepthwiseConv2D 'groups' parameter issue
        def custom_depthwise_conv2d(**kwargs):
            kwargs.pop('groups', None)  # Remove invalid 'groups' parameter
            return tf.keras.layers.DepthwiseConv2D(**kwargs)
        
        # Register custom object to handle model loading issues
        tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = custom_depthwise_conv2d
        
        model = tf.keras.models.load_model("model/keras_model.h5")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"🚨 Model failed to load:\n {e}")
        st.info("💡 **Troubleshooting Tips:**\n"
                "- Check if 'model/keras_model.h5' exists\n"
                "- Verify TensorFlow/Keras version compatibility\n"
                "- Try re-exporting the model from Teachable Machine")
        return None

model = load_model()

# Define your class names here in the same order as your model's output
class_names = ["Normal", "Anomalous"]

st.title("🔍 Anomaly Detection with Teachable Machine")
st.write("Upload or capture an image to detect if it's **normal** or **anomalous**.")

# Add model status indicator
if model is not None:
    st.success("🤖 **Model Status:** Ready for predictions")
else:
    st.error("🤖 **Model Status:** Failed to load")

# User selects input method
input_method = st.radio("📷 **Select image input method:**", ("Upload Image", "Capture Image"))

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📤 Uploaded Image", use_column_width=True)
            
            # Show image info
            st.info(f"📋 **Image Info:** {img.size[0]}×{img.size[1]} pixels, {uploaded_file.type}")
        except Exception as e:
            st.error(f"❌ **Invalid image file:** {e}")
            
elif input_method == "Capture Image":
    captured_img = st.camera_input("📸 Take a picture")
    if captured_img is not None:
        try:
            img = Image.open(captured_img).convert("RGB")
            st.image(img, caption="📷 Captured Image", use_column_width=True)
            
            # Show image info
            st.info(f"📋 **Image Info:** {img.size[0]}×{img.size[1]} pixels")
        except Exception as e:
            st.error(f"❌ **Invalid image capture:** {e}")

# If image and model are available, make prediction
if img is not None and model is not None:
    try:
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔄 Analyzing image..."):
            predictions = model.predict(img_array, verbose=0)  # Suppress prediction logs
            pred_idx = int(np.argmax(predictions))
            confidence = float(np.max(predictions) * 100)
            
            # Display prediction with styling
            if confidence > 80:
                confidence_color = "🟢"
            elif confidence > 60:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"
            
            st.write(f"### ✅ Prediction: **{class_names[pred_idx]}** {confidence_color}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Add interpretation
            if class_names[pred_idx] == "Anomalous" and confidence > 70:
                st.warning("⚠️ **Anomaly detected with high confidence!**")
            elif class_names[pred_idx] == "Normal" and confidence > 80:
                st.success("✅ **Normal pattern detected.**")
            elif confidence < 60:
                st.info("🤔 **Low confidence prediction.** Consider using a clearer image.")

            # Display confidence for each class
            if predictions.shape[1] == len(class_names):
                st.write("### 📊 Class Probabilities:")
                df = pd.DataFrame(predictions[0] * 100, index=class_names, columns=["Confidence (%)"])
                st.bar_chart(df)
                
                # Show raw probabilities
                with st.expander("🔍 View Raw Probabilities"):
                    for i, class_name in enumerate(class_names):
                        st.write(f"**{class_name}:** {predictions[0][i]:.4f}")
        
    except Exception as e:
        st.error(f"❌ **Prediction failed:** {e}")
        st.info("💡 **Suggestions:**\n"
                "- Try a different image format (JPG/PNG)\n"
                "- Ensure image is not corrupted\n"
                "- Check if model expects different input size")

elif img is not None and model is None:
    st.error("❌ **Cannot make prediction:** Model not loaded")
    
elif model is None:
    st.warning("⚠️ **Model not loaded.** Please check the model path and try refreshing the page.")
    
    # Add model info section
    with st.expander("ℹ️ Model Information"):
        st.write("""
        **Expected Model Format:** Teachable Machine Keras model (.h5)
        **Input Size:** 224x224 pixels
        **Classes:** Normal, Anomalous
        **Model Path:** `model/keras_model.h5`
        """)
        
        st.write("**To fix model loading issues:**")
        st.code("""
        # Ensure your directory structure is:
        your_app/
        ├── app.py
        └── model/
            └── keras_model.h5
        """)

# Add footer with additional info
st.markdown("---")
st.markdown("**💡 Tips for better results:**")
st.markdown("- Use clear, well-lit images")
st.markdown("- Ensure the subject is clearly visible")
st.markdown("- Try different angles if confidence is low")
st.markdown("- Images should be similar to training data")