# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your TensorFlow model
model = load_model('C:/Users/pc/Downloads/Dog-vs-Cat-Classification-Model/dog_cat_model')

def predict_image_class(input_image):
    # Resize, scale, and reshape the input image
    input_image_resize = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resize / 255
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    # Make predictions using the model
    input_prediction = model.predict(image_reshaped)

    # Get the predicted label
    input_pred_label = np.argmax(input_prediction)

    return input_pred_label




# Streamlit app
st.title("Dog vs. Cat Image Classification with TensorFlow")
st.write("This project demonstrates a simple Streamlit app for image classification using a pre-trained TensorFlow model.")

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    input_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions when the user clicks the "Classify" button
    if st.button("Classify"):
        # Predict the image class
        pred_label = predict_image_class(input_image)
        

        # Display the result based on the predicted label
        if pred_label == 0:
         st.success("The image represents a Cat")
        else:
         st.success("The image represents a Dog")
