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


transforms = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor()
])

# Модель UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder = models.vgg11_bn(pretrained=True).features[:17]
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x

@st.cache_resource()
def get_model():
    # Инициализация модели UNet
    unet_model = UNet()
    # Загрузка весов в модель
    state_dict = torch.load("model_forest_unet/forest.pth", map_location=torch.device('cpu'))
    unet_model.load_state_dict(state_dict)
    unet_model.eval()  # Переключение модели в режим оценки

    return unet_model

unet_model = get_model()

st.title("UNet for Image Segmentation")

url = st.text_input("Enter image URL")
# Загрузка изображения
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        uploaded_file=image
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")


if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        image = uploaded_file
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Segment Image"):
        with st.spinner('Processing...'):
            # Преобразование изображения для модели
            input_image = T.Compose([
                T.Resize(256),  # Адаптируйте размер под вашу модель UNet
                T.ToTensor(),
            ])(image).unsqueeze(0)

            # Получение предсказания от модели
            with torch.no_grad():
                output = unet_model(input_image)
                # Преобразование выходных данных модели обратно в изображение
                output_image = output.squeeze().cpu().numpy()
                st.image(output_image, caption="Segmented Image", use_column_width=True)
