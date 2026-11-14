import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

# --------------------------------------------------
# Configuration — preserve the existing model/classes
# --------------------------------------------------
MODEL_PATH = "model_custom_cnn.h5"
IMG_HEIGHT = 128
IMG_WIDTH = 128

CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

st.set_page_config(
    page_title="BioPrint Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Clinical research theme
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F9F8;
        color: #183B38;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 249, 248, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E0EBE8;
    }

    .brand {
        font-size: 1.7rem;
        font-weight: 700;
        color: #126D63;
        letter-spacing: -0.8px;
    }

    .subtitle {
        color: #718783;
        font-size: 0.88rem;
        margin-top: -8px;
    }

    .section-title {
        color: #183B38;
        font-size: 1.18rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #718783;
        font-size: 0.9rem;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #DFEAE7;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 14px;
    }

    .result-label {
        color: #718783;
        font-size: 0.83rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .result-value {
        color: #126D63;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.25;
    }

    .status-pill {
        display: inline-block;
        background: #E3F4EF;
        color: #126D63;
        border-radius: 30px;
        padding: 6px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .notice {
        background: #FFF8E8;
        color: #795A20;
        border: 1px solid #F0E1B9;
        padding: 13px 15px;
        border-radius: 10px;
        font-size: 0.85rem;
    }

    div.stButton > button[kind="primary"] {
        background: #147D70;
        border: 1px solid #147D70;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        min-height: 46px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #10685E;
        border-color: #10685E;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #A9C9C1;
        border-radius: 12px;
        padding: 10px;
    }

    hr {
        border-color: #E0EBE8;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource
def load_prediction_model():
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file.resolve()}"
        )

    return load_model(model_file, compile=False)


# --------------------------------------------------
# Existing image preprocessing and TTA
# Keep the original transformations and class order
# --------------------------------------------------
def predict_fingerprint(image, model):
    images_to_predict = []

    # 1. Original image
    images_to_predict.append(image)

    # 2. Horizontal flip
    images_to_predict.append(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

    # 3. Rotate +10 degrees
    images_to_predict.append(image.rotate(10))

    # 4. Rotate -10 degrees
    images_to_predict.append(image.rotate(-10))

    # 5. Rotate +5 degrees
    images_to_predict.append(image.rotate(5))

    batch_images = []

    for img_var in images_to_predict:
        img_resized = img_var.resize((IMG_HEIGHT, IMG_WIDTH))
        img_arr = np.array(img_resized.convert("L"))
        img_arr = np.expand_dims(img_arr, axis=-1)
        img_arr = img_arr / 255.0
        batch_images.append(img_arr)

    batch_images = np.array(batch_images, dtype=np.float32)

    predictions = model.predict(batch_images, verbose=0)
    avg_prediction = np.mean(predictions, axis=0)

    return avg_prediction


# --------------------------------------------------
# Session state: recent predictions
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">BioPrint</div>'
        '<div class="subtitle">FINGERPRINT RESEARCH PLATFORM</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Project overview")
    st.write("Fingerprint-based blood-group classification prototype.")

    st.markdown("### Model information")
    st.write("**Architecture:** Custom CNN")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input:** 128 × 128 grayscale")
    st.write("**Output classes:** 8")
    st.write("**Inference:** Five image variations averaged")

    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a section",
        ["Fingerprint Analysis", "Prediction History", "About the Model"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Runs locally using the saved model.")
    st.caption("Research prototype — not for medical decisions.")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="brand">Fingerprint Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Analyze a fingerprint image using the existing eight-class CNN model.'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------
# Page: Fingerprint Analysis
# --------------------------------------------------
if page == "Fingerprint Analysis":

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">01 · Upload fingerprint</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="muted">'
            'Choose a fingerprint image from your computer.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        uploaded_file = st.file_uploader(
            "Select an image",
            type=["bmp", "jpg", "jpeg", "png"],
            help="Supported formats: BMP, JPG, JPEG and PNG.",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).copy()

                st.image(
                    image,
                    caption="Uploaded fingerprint",
                    use_container_width=True,
                )

                st.caption(
                    f"Original image dimensions: "
                    f"{image.width} × {image.height} pixels"
                )

                if st.button(
                    "Analyze fingerprint",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Running CNN inference..."):
                            model = load_prediction_model()
                            avg_prediction = predict_fingerprint(image, model)

                        if (
                            avg_prediction.shape != (len(CLASS_NAMES),)
                            or not np.isfinite(avg_prediction).all()
                        ):
                            st.error(
                                "The model returned unexpected output. "
                                "No result was saved."
                            )
                        else:
                            class_idx = int(np.argmax(avg_prediction))
                            result = CLASS_NAMES[class_idx]
                            score = float(avg_prediction[class_idx])

                            # Save the real output; do not invent results.
                            record = {
                                "Time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "Image": uploaded_file.name,
                                "Top output class": result,
                                "Top-class score": f"{score:.2%}",
                            }

                            st.session_state.latest_result = {
                                "image_name": uploaded_file.name,
                                "class_name": result,
                                "score": score,
                                "probabilities": avg_prediction.tolist(),
                                "time": record["Time"],
                            }

                            st.session_state.prediction_history.insert(0, record)
                            st.session_state.prediction_history = (
                                st.session_state.prediction_history[:20]
                            )

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

            except Exception as exc:
                st.error(f"Could not open this image: {exc}")

        else:
            st.markdown(
                '<div class="panel">'
                '<div class="section-title">Ready for analysis</div>'
                '<div class="muted">'
                'Your selected image will appear here before inference.'
                '</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            '<div class="section-title">02 · Analysis results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="muted">'
            'Results from the most recent analysis in this session.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        latest = st.session_state.latest_result

        if latest is None:
            st.markdown(
                '<div class="panel">'
                '<div class="result-label">Predicted class output</div>'
                '<div class="result-value">—</div>'
                '<p class="muted">'
                'Upload an image and select Analyze fingerprint to see '
                'the model output.'
                '</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="panel">'
                f'<div class="result-label">Top model output</div>'
                f'<div class="result-value">{latest["class_name"]}</div>'
                f'<div class="muted">Image: {latest["image_name"]}</div>'
                f'<hr>'
                f'<div class="result-label">Top-class output score</div>'
                f'<h3>{latest["score"]:.2%}</h3>'
                f'<div class="muted">'
                f'Time: {latest["time"]}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Output probabilities")

            probability_map = {
                label: float(value)
                for label, value in zip(
                    CLASS_NAMES, latest["probabilities"]
                )
            }

            st.bar_chart(
                probability_map,
                horizontal=True,
                x_label="Model output",
                y_label="Class",
            )

            st.markdown("#### How to interpret this result")

            st.write(
                f"The model assigned its highest output to **"
                f"{latest['class_name']}** for this image."
            )

            st.write(
                f"The corresponding output score is "
                f"**{latest['score']:.2%}**. This describes the model's "
                "output, not a verified probability that the person's "
                "actual blood group is correct."
            )

            st.caption(
                "The app averages outputs from the original image and "
                "four transformed versions. This is not an independent "
                "validation or a medical test."
            )

    st.write("")
    st.markdown(
        '<div class="notice">'
        '<strong>Research disclaimer:</strong> This prototype has not '
        'been established as a reliable method for determining blood '
        'group. Confirm blood group using an appropriate laboratory test. '
        'Do not use these outputs for medical decisions.'
        '</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Page: Prediction History
# --------------------------------------------------
elif page == "Prediction History":
    st.markdown("### Recent predictions")
    st.caption(
        "History is kept in the current Streamlit session only. "
        "It is not saved to a permanent database."
    )

    history = st.session_state.prediction_history

    if history:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear history"):
            st.session_state.prediction_history = []
            st.session_state.latest_result = None
            st.rerun()
    else:
        st.info(
            "No predictions yet. Run an analysis to populate this section."
        )

# --------------------------------------------------
# Page: About the Model
# --------------------------------------------------
elif page == "About the Model":
    st.markdown("### Model and processing")

    st.markdown(
        """
        This prototype loads the existing saved Custom CNN checkpoint.

        **Image processing**
        - Resize each image variation to 128 × 128 pixels.
        - Convert each variation to grayscale.
        - Scale pixel values by dividing by 255.
        - Create a batch containing five image variations.

        **Test-time augmentation**
        - Original image
        - Horizontal flip
        - Rotation by +10 degrees
        - Rotation by -10 degrees
        - Rotation by +5 degrees

        **Output**
        - The model produces eight class outputs.
        - The app averages the five outputs.
        - The class with the highest averaged output is displayed.
        """
    )

    st.warning(
        "The interface does not establish the provenance of dataset labels, "
        "independent test accuracy, or clinical validity. Those require "
        "separate evidence and evaluation."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "BioPrint Research · Local student prototype · "
    "TensorFlow / Streamlit"
)
