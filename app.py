import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Load Both Models
# -----------------------------
clf_model = tf.keras.models.load_model("bird_drone_cnn.keras")   # Classification Model
det_model = YOLO("best.pt")                                       # YOLO Detection Model

# Class names
class_names = ["Bird", "Drone"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Bird vs Drone - Classification & Detection 🐦🚁")

mode = st.sidebar.radio("Choose Mode", ["Classification", "Detection"])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # 1️⃣ CLASSIFICATION
    # -----------------------------
    if mode == "Classification":
        st.subheader("Classification Result")

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = clf_model.predict(img_array)
        result = np.argmax(prediction)

        st.write(f"### Prediction: **{class_names[result]}**")

        st.write("### Confidence Scores:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {prediction[0][i]*100:.2f}%")

    # -----------------------------
    # 2️⃣ DETECTION
    # -----------------------------
    if mode == "Detection":
        st.subheader("YOLO Detection Result")

        results = det_model(image)
        result_img = results[0].plot()  # draw bounding boxes

        st.image(result_img, caption="Detected Image", use_column_width=True)
