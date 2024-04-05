from urllib import response
import streamlit as st

import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import sys
from model_cosplay.yolov9.detect import run

def increase_resolution_in_place(image_path, target_width):
    image = Image.open(image_path)
    imgwidth=image.size[0]
    imgheight=image.size[1]
    targetheight = int(imgheight * (target_width / imgwidth))
    high_res_image = image.resize((target_width, targetheight), Image.LANCZOS)
    high_res_image.save(image_path)

def increase_resolution_pic(image, target_width):
    imgwidth=image.size[0]
    imgheight=image.size[1]
    targetheight = int(imgheight * (target_width / imgwidth))
    high_res_image = image.resize((target_width, targetheight), Image.LANCZOS)
    return high_res_image

def getweights():
    #if file not exist
    if not os.path.exists("model_cosplay/cosplay.pt"):
        os.system("""
                cd model_cosplay
                cat cosplay_patr_aa cosplay_patr_ab > cosplay.pt
                cd ..
                """) 
    return "model_cosplay/cosplay.pt"

#количество колонок
options = ["одна колонка", "две колонки"]
selected_model = st.sidebar.radio("режим отображения", options)
#уверенность модели
confidence = st.sidebar.slider("порог уверенности модели", 0.0, 1.0, 0.25) 
#iou
iou = st.sidebar.slider("допустимое пересечение боксов (iou)", 0.0, 1.0, 0.3)
#agnostic nms
agnostic_nms = st.sidebar.checkbox("agnostic nms (один объект - один бокс)", True)



def predict(img):
    col1, col2 = st.columns(2) if selected_model == "две колонки" else st.columns(1) +[st.empty()]
    with col1:
        img_pil = Image.open(img).convert("RGB")
        img_pil.save("img.jpg")
        increase_resolution_in_place("img.jpg", 1024)
        img_np = cv2.imread("img.jpg")[:, :, ::-1]
        st.image(img_np, use_column_width=True)

    with col2 if selected_model == "две колонки" else col1:
        with st.spinner('Processing...'):
            out_path = run(source = "img.jpg",
                            weights= getweights(),
                            conf_thres= confidence,
                            iou_thres=iou,
                            agnostic_nms=agnostic_nms)
            st.image(open(str(out_path)+"/img.jpg", 'rb').read(), use_column_width=True)

    os.remove("img.jpg")
    os.remove(str(out_path)+"/img.jpg")
    os.removedirs ( str(out_path))




st.title("Cosplay Detection")
st.write("""
    Upload an image you want to detect cosplay objects on or enter an image URL.
    """)
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
url = st.text_input("Enter image URL")



if url:  
    try:
        response = requests.get(url)
        if response.status_code == 200: 
            img = BytesIO(response.content) 
            predict(img)
        else:
            st.warning("Invalid URL. Make sure the URL is correct and the image exists.")
    except requests.exceptions.MissingSchema:
        st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")

if uploaded_files is not None:
    for img in uploaded_files:
        
        predict(img)
