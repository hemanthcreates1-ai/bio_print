import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import ResNet50
import numpy as np
from PIL import Image

# Configuration
MODEL_PATH = 'model_custom_cnn.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Function to load model
# Note: Since the saved model might have custom objects or structure issues 
# when loading directly if not saved as a full model, we might need to recreate architecture.
# However, train.py saved the full model, so load_model should work.
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Page Config
st.set_page_config(page_title="Blood Group Detector", page_icon="🩸", layout="centered")

# Sidebar
st.sidebar.title("Project Details")
st.sidebar.info(
    """
    **Project Name:** Fingerprint-Based Blood Group Detection
    \n**Model:** Custom CNN (Optimized for Fingerprints)
    \n**Framework:** TensorFlow / Keras
    \n**Status:** Production Ready
    """
)

# Main Content
st.title("🩸 Blood Group Detection System")
st.write("Upload a **fingerprint image** to detect the blood group.")

uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["bmp", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Fingerprint', use_column_width=True)
    
    # Predict Button
    if st.button('🔍 Identify Blood Group'):
        with st.spinner('Analyzing fingerprint pattern...'):
            model = load_prediction_model()
            
            if model:
                # Test Time Augmentation (TTA)
                # Create variations of the image to average predictions
                images_to_predict = []
                
                # 1. Original
                images_to_predict.append(image)
                # 2. Horizontal Flip
                images_to_predict.append(image.transpose(Image.FLIP_LEFT_RIGHT))
                # 3. Rotate +10
                images_to_predict.append(image.rotate(10))
                # 4. Rotate -10
                images_to_predict.append(image.rotate(-10))
                # 5. Rotate +5
                images_to_predict.append(image.rotate(5))

                batch_images = []
                for img_var in images_to_predict:
                     # Preprocess
                    img_resized = img_var.resize((IMG_HEIGHT, IMG_WIDTH))
                    img_arr = np.array(img_resized.convert('L'))
                    img_arr = np.expand_dims(img_arr, axis=-1)
                    img_arr = img_arr / 255.0
                    batch_images.append(img_arr)
                
                batch_images = np.array(batch_images)
                
                # Predict on batch
                predictions = model.predict(batch_images)
                
                # Average probabilities
                avg_prediction = np.mean(predictions, axis=0)
                
                class_idx = np.argmax(avg_prediction)
                confidence = float(np.max(avg_prediction))
                
                result = CLASS_NAMES[class_idx]
                
                # Display Result
                st.success(f"## Predicted Blood Group: {result}")
                st.progress(confidence)
                st.write(f"**Confidence Score:** {confidence:.2%}")
                
                # Disclaimer
                st.warning("Note: This is an AI-based prediction for educational purposes.")
