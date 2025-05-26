import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load model
load_model = tf.keras.models.load_model
model = load_model("mnist_digit_model.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload an image with one or more handwritten digits (e.g., '13540').")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def resize_and_pad(image):
    h, w = image.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_and_segment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10 and w*h > 150:
            digit_img = thresh[y:y+h, x:x+w]
            processed_digit = resize_and_pad(digit_img) / 255.0
            digits.append(processed_digit.reshape(28, 28, 1))
            bboxes.append((x, y, w, h))

    sorted_digits = [x for _, x in sorted(zip(bboxes, digits), key=lambda b: b[0][0])]
    return sorted_digits

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)

    st.image(img, caption="Original Image", width=250)

    digits = preprocess_and_segment(img_np)

    if digits:
        st.subheader("Predicted Digits:")
        col1, col2 = st.columns(2)
        predictions = ""

        for i, digit in enumerate(digits):
            prediction = model.predict(np.expand_dims(digit, axis=0))
            predicted_digit = np.argmax(prediction)

            with col1:
                st.image(digit.squeeze(), width=50, caption=f"Digit {i+1}")
            with col2:
                st.write(f"Prediction: {predicted_digit}")

            predictions += str(predicted_digit)

        st.success(f"Full Prediction: {predictions}")
    else:
        st.warning("No digits detected. Use clear black digits on a white background.")
