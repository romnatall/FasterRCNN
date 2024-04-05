import streamlit as st
from PIL import Image
st.title("FRCNN project")

description_show_options = ['main','UNet forest model','Turbine model','Vegetation model','Cosplay model (bonus)']
description_show = st.sidebar.radio("Description", description_show_options)

if description_show == 'Cosplay model (bonus)':
    st.write("Эта модель обучена для обнаружения косплееров на изображениях.")

    st.write("Модель использует архитектуру YOLO9 для обнаружения косплееров на изображениях.") 
    st.write("Для обучения модели использовался датасет из 200 изображений косплееров. После применения аугментации данных этот набор был увеличен до 600 изображений.")
    st.write("Модель обучалась в течение 25 эпох.")

    st.subheader("Пример изображений с обнаруженными косплеерами:")
    col1, col2 = st.columns(2)
    with col1:
        cosplayer_image1 = Image.open("images/exp4/img1.jpg")
        st.image(cosplayer_image1, caption="Пример с обнаруженными косплеерами 1", use_column_width=True)
    with col2:
        cosplayer_image2 = Image.open("images/exp4/img2.jpg")
        st.image(cosplayer_image2, caption="Пример с обнаруженными косплеерами 2", use_column_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        cosplayer_image1 = Image.open("images/exp4/train_batch0.jpg")
        st.image(cosplayer_image1, caption="Пример 1", use_column_width=True)
    with col2:
        cosplayer_image2 = Image.open("images/exp4/train_batch1.jpg")
        st.image(cosplayer_image2, caption="Пример 2", use_column_width=True)
    with col3:
        cosplayer_image3 = Image.open("images/exp4/train_batch2.jpg")
        st.image(cosplayer_image3, caption="Пример 3", use_column_width=True)
    
    st.subheader("Метрики модели:")
    
    st.write("- Матрица ошибок (Confusion Matrix):")
    confusion_matrix_image = Image.open("images/exp4/confusion_matrix.png")
    st.image(confusion_matrix_image, caption="Матрица ошибок", use_column_width=True)
    
    st.write("- График Precision-Recall:")
    pr_curve_image = Image.open("images/exp4/PR_curve.png")
    st.image(pr_curve_image, caption="Precision-Recall график", use_column_width=True)

    st.write("- График F1-Score:")
    f1_curve_image = Image.open("images/exp4/F1_curve.png")
    st.image(f1_curve_image, caption="F1-Score график", use_column_width=True)
    
    st.write("- График R-Curve:")
    r_curve_image = Image.open("images/exp4/R_curve.png")
    st.image(r_curve_image, caption="R-Curve график", use_column_width=True)
    
    st.write("- График P-Curve:")
    p_curve_image = Image.open("images/exp4/P_curve.png")
    st.image(p_curve_image, caption="P-Curve график", use_column_width=True)
    
    st.write("- График результатов:")
    results_image = Image.open("images/exp4/results.png")
    st.image(results_image, caption="График результатов", use_column_width=True)
    
    st.write("- Таблица гиперпараметров (Hyp.yaml):")
    with open("images/exp4/hyp.yaml", "r") as hyp_file:
        hyp_contents = hyp_file.read()
    st.code(hyp_contents)

    st.write("- Таблица оптимизаций (Opt.yaml):")
    with open("images/exp4/opt.yaml", "r") as opt_file:
        opt_contents = opt_file.read()
    st.code(opt_contents)

if description_show == 'Turbine model':
    st.header("Описание модели с ветряками и столбами")
    st.write("Модель использует архитектуру YOLOv8 для обнаружения ветряков и электрических столбов на изображениях.")
    st.write("Для обучения модели использовался датасет из 2885 изображений, содержащих ветряки и электрические столбы.")
    st.write("Модель обучалась в течение 8 эпох.")
    st.subheader("Пример изображений с обнаруженными ветряками и столбами:")
    col1, col2 = st.columns(2)
    with col1:
        turbine_image1 = Image.open("images/yolo results/img1.jpg")
        st.image(turbine_image1, caption="Пример с обнаруженными ветряками 1", use_column_width=True)
    with col2:
        turbine_image2 = Image.open("images/yolo results/img2.jpg")
        st.image(turbine_image2, caption="Пример с обнаруженными ветряками 2", use_column_width=True)
    st.subheader("Метрики модели:")

    st.write("- Матрица ошибок (Confusion Matrix):")
    confusion_matrix_normalized_image = Image.open("images/yolo results/confusion_matrix_normalized.png")
    st.image(confusion_matrix_normalized_image, caption="Нормализованная матрица ошибок", use_column_width=True)
    
    st.write("- График Precision-Recall:")
    pr_curve_image = Image.open("images/yolo results/PR_curve (1).png")
    st.image(pr_curve_image, caption="Precision-Recall график", use_column_width=True)

    st.write("- График результатов:")
    results_image = Image.open("images/yolo results/results (1).png")
    st.image(results_image, caption="График результатов", use_column_width=True)

elif description_show == 'Vegetation model':

    st.write("Модель на основе ResNet-18 была разработана для классификации и локализации объектов на изображениях. В этой модели слои извлекателя признаков (backbone) ResNet-18 были заморожены, чтобы сохранить предварительно обученные веса, а затем добавлены отдельные блоки для классификации и предсказания координат объектов.")
    st.write("Модель была обучена в течение 10 эпох.")

    st.header("Датасет")
    st.write("Датасет состоял из 118 изображений, содержащих огурцы, баклажаны и грибы. Этот небольшой размер датасета может представлять сложности для обучения модели из-за ограниченного количества данных.")

    st.header("Ключевые особенности")
    st.write("- Использование модели ResNet-18 для классификации и локализации объектов.")
    st.write("- Обучение модели на небольшом датасете в течение 10 эпох.")
    st.write("- Применение замороженных слоев извлекателя признаков и отдельных блоков для классификации и предсказания координат.")


elif description_show == 'UNet model':
    pass


