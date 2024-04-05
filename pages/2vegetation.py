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
trnsfrms2 = T.Compose([
    T.Resize((299, 299)),  # Измените размер в соответствии с требованиями вашей модели
    T.ToTensor()
])


class LocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.clf = nn.Sequential(
            nn.Linear(512 * 8 * 8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        self.box = nn.Sequential(
            nn.Linear(512 * 8 * 8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        logits = self.clf(resnet_out)
        box_coords = self.box(resnet_out)
        return logits, box_coords

class_names = ['cucumber', 'eggplant', 'mushroom']
model = torch.load("model_vegetation/veg.pth", map_location=torch.device('cpu'))
model.eval()
st.title("This App recognizes vegetables in images")

url = st.text_input("Enter image URL")

def detect(image: Image):
    img = image.convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),  # Адаптируйте размер под вашу модель
        T.ToTensor(),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    logits, box_coords = model(img_tensor)

    # Обработка результатов классификации
    # Получаем предсказанный класс
    _, predicted_class = torch.max(logits, 1)
    predicted_class = predicted_class.item()  # Преобразуем к скаляру

    # Преобразуем координаты в NumPy массив для визуализации
    coords = box_coords.detach().cpu().numpy()
    box_coords = coords[0]  # Предполагаем, что у нас есть только один объект на изображении

    # Отрисовка прямоугольника вокруг обнаруженного объекта
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([(box_coords[0]*img.width, box_coords[1]*img.height),
                    (box_coords[2]*img.width, box_coords[3]*img.height)], outline="red", width=3)
    
    # Добавляем метку класса
    draw.text((box_coords[0]*img.width, box_coords[1]*img.height),
            f"{class_names[predicted_class]}", fill="red")

    return img_draw, predicted_class

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
        
# Загрузка и обработка изображения из файла
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png" ,"webp"])

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner('Processing...'):
        img, predicted_class = detect(image)
        st.image(img, caption="Processed Image", use_column_width=True)
