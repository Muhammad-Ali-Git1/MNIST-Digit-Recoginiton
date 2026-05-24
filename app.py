import streamlit as st
import numpy as np
import cv2
import joblib

st.title("MNIST Digit Recognizer (MLP)")

ml = joblib.load("digit_model.pkl")

uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:


    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))

    img = img / 255.0
    if img.mean() > 0.5:
        img = 1 - img

    img = (img > 0.2).astype(np.float32)
    img_input = img.reshape(1, -1)
    prediction = ml.predict(img_input)
    st.image(img, caption="Processed Image", width=150)

    st.success(f"Predicted Digit: {prediction[0]}")
