import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("bird_drone_cnn.keras")

# Class names
class_names = ["Bird", "Drone"]

st.title("Bird vs Drone Classification 🐦🚁")
st.write("Upload an image and the model will classify it as Bird or Drone.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    st.subheader("Prediction:")
    st.write(f"**{class_names[result]}**")

    st.subheader("Confidence Scores:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")





