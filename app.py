import streamlit as st
import numpy as np
import cv2
import joblib

st.title("MNIST Digit Recognizer (MLP)")

ml = joblib.load("digit_model.pkl")

uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # resize to MNIST size
    img = cv2.resize(img, (28, 28))

    # normalize
    img = img / 255.0

    # 🔥 FIX 1: invert image (MNIST-style)
    if img.mean() > 0.5:
        img = 1 - img

    # 🔥 FIX 2: threshold to remove noise
    img = (img > 0.2).astype(np.float32)

    # flatten for MLP
    img_input = img.reshape(1, -1)

    # predict
    prediction = ml.predict(img_input)

    # show processed image
    st.image(img, caption="Processed Image", width=150)

    st.success(f"Predicted Digit: {prediction[0]}")