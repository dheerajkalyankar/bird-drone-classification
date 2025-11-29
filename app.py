import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    clf_model = tf.keras.models.load_model("bird_drone_cnn.keras")  # Classification model
    det_model = YOLO("best.pt")                                       # YOLO detection model
    return clf_model, det_model

clf_model, det_model = load_models()

# -----------------------------
# Preprocess function for classification
# -----------------------------
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# YOLO detection function
# -----------------------------
def detect_objects(img: Image.Image):
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = det_model(img_bgr)

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{det_model.names[int(cls)]} {score:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# -----------------------------
# Streamlit interface
# -----------------------------
st.title("Bird vs Drone Analyzer 🐦🚁")
task = st.radio("Choose Task:", ["Classification", "Detection"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=400)

    if task == "Classification":
        arr = preprocess_image(img)
        pred = clf_model.predict(arr)[0][0]
        label = "Drone" if pred >= 0.5 else "Bird"
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {pred*100:.2f}%")

    else:
        detected_img = detect_objects(img)
        st.image(detected_img, caption="Detected Objects", width=500)
