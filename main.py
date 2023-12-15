import base64
import streamlit as st
import numpy as np
from PIL import ImageOps, Image
import keras
import requests 
from io import BytesIO
from util import classify, set_background


# Load classifier
model = keras.models.load_model("./model/Drowness_tuned.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Set background
set_background('./bgs/bgr.png')

# Set title
st.title('Drowsiness & yawn detection')

# Set header
st.header('Please upload a chest X-ray image or provide an image URL')

# Upload file or get image URL
file = st.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])
image_url = st.text_input("Or paste an image URL")

# Display image and make prediction
if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        class_name, conf_score = classify(image, model, class_names)
        st.write("## Prediction: {}".format(class_name))
        st.write("### Confidence: {}%".format(int(conf_score * 100)))
    except Exception as e:
        st.error(f"Error loading image: {e}")
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(image, caption='Image from URL', use_column_width=True)
        class_name, conf_score = classify(image, model, class_names)
        st.write("## Prediction: {}".format(class_name))
        st.write("### Confidence: {}%".format(int(conf_score * 100)))
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
