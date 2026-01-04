import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Autism & Emotion Detection",
    page_icon="🧠",
    layout="wide"
)

MODELS_DIR = "models"

# =============================
# LOAD MODELS (LOCAL .keras)
# =============================
@st.cache_resource
def load_models():
    try:
        autism_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "autism_model.keras")
        )
        emotion_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "emotion_model.keras")
        )

        with open(os.path.join(MODELS_DIR, "emotion_classes.json"), "r") as f:
            emotion_classes = json.load(f)

        # index → label
        emotion_classes = {v: k for k, v in emotion_classes.items()}

        return autism_model, emotion_model, emotion_classes

    except Exception as e:
        st.error("❌ Failed to load models")
        st.exception(e)
        return None, None, None


# =============================
# IMAGE PREPROCESSING
# =============================
def preprocess_image(image, target_size=(150, 150)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


# =============================
# MAIN APP
# =============================
def main():
    st.title("🧠 Autism & Emotion Detection System")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ System Info")
    st.sidebar.info(f"TensorFlow: {tf.__version__}")
    st.sidebar.info(f"Keras: {tf.keras.__version__}")

    detection_type = st.sidebar.selectbox(
        "Detection Mode",
        ["Autism Detection", "Emotion Recognition", "Both"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **How to use**
    1. Upload a face image
    2. Click Analyze
    3. View results
    """)

    autism_model, emotion_model, emotion_classes = load_models()

    if autism_model is None:
        st.stop()

    col1, col2 = st.columns(2)

    # =============================
    # IMAGE UPLOAD
    # =============================
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    # =============================
    # ANALYSIS
    # =============================
    with col2:
        st.header("📊 Results")

        if uploaded_file and st.button("🔍 Analyze", use_container_width=True):
            img = preprocess_image(image)

            # Autism Detection
            if detection_type in ["Autism Detection", "Both"]:
                st.subheader("🧩 Autism Detection")
                pred = autism_model.predict(img, verbose=0)[0][0]
                label = "Autistic" if pred > 0.5 else "Non-Autistic"
                confidence = pred if pred > 0.5 else 1 - pred

                if label == "Autistic":
                    st.error(label)
                else:
                    st.success(label)

                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.progress(float(confidence))

            # Emotion Recognition
            if detection_type in ["Emotion Recognition", "Both"]:
                st.subheader("😊 Emotion Recognition")
                preds = emotion_model.predict(img, verbose=0)[0]
                idx = np.argmax(preds)
                label = emotion_classes.get(idx, "Unknown")

                st.info(f"Emotion: **{label}**")
                st.metric("Confidence", f"{preds[idx]*100:.2f}%")

                st.markdown("**All probabilities:**")
                for i, p in enumerate(preds):
                    st.write(f"- {emotion_classes.get(i)}: {p*100:.2f}%")

        if not uploaded_file:
            st.info("👆 Upload an image to start")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:12px'>"
        "Powered by TensorFlow • Streamlit • Keras 3"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
