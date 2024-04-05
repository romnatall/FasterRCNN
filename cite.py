import streamlit as st

st.title("FRCNN project")

description_show_options = ['Cosplay model','Turbine model','Vegetation model','UNet model']
description_show = st.sidebar.radio("Description", description_show_options)

if description_show == 'Cosplay model':
    pass
elif description_show == 'Turbine model':
    pass
elif description_show == 'Vegetation model':
    pass
elif description_show == 'UNet model':
    pass


