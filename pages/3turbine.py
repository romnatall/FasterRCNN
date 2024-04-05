import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from ultralytics import YOLO
import numpy as np
import torch.nn as nn
from torchvision import models

# Определяем трансформации изображения


class_names = ['cable tower', 'turbine']  # Классы, определенные вашей моделью

@st.cache_resource()
def get_model():
    model = YOLO("model_turbines/turbines.pt")  # Загрузите вашу модель YOLO
    return model


st.title("This App recognizes turbines and cabel towers in images")

url = st.text_input("Enter image URL")

def detect(image: Image) -> Image:
    img = image.convert("RGB")
    img_array = np.array(img)
    model =  get_model()
    detect_results = model(img_array)
    detect_img = detect_results[0].plot()
    detect_img = Image.fromarray(detect_img)
    return detect_img


# Загрузка и обработка изображения из файла
uploaded_files = st.file_uploader("Or upload an image file",allow_multiple_files=True, type=["jpg", "jpeg", "png" ,"webp"])

if url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify URL Image"):
            with st.spinner('Processing...'):
                processed_image = detect(image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Classify Uploaded Image"):
        with st.spinner('Processing...'):
            processed_image = detect(image)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
