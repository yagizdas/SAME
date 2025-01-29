import streamlit as st
from PIL import Image
import numpy as np
from SAME.SAME import SlidingWindowObjectDetection as SWOD
from torch import jit
from SAME.device_detect import DeviceDetermination

device = DeviceDetermination()

device = device()


st.set_page_config(layout="wide")
st.title(":earth_africa: **SAME** :earth_americas:")
st.caption("*Personal Project by Kemal Yağız Daşkıran*")    
st.header(":blue[S]atellite & :blue[A]erial :blue[M]apping :blue[E]ngine!   :airplane:", divider=True)
st.caption("To start, please upload an aerial or satellite imagery to analyze into map regions.")


col1,col2, col3 = st.columns((1.2,1,2.2))
col1.write("#")
imgpath = col1.file_uploader(":world_map: Please choose an imagery to Map!", type=['jpeg', 'jpg'])

if imgpath is not None:
    img = Image.open(imgpath)
    ph = st.empty()
    img_array = np.array(img)
    h_w = [img_array.shape[1],img_array.shape[0]]

    preset_array = min(h_w)

    preset_choice = col2.radio(
    ":gear: - Please select the mapping range: ",
    ["Large", "Medium", "Small"],
    captions=[
        "***Recommended***",
        "",
        "***Detailed, but more prone to Errors***",
    ])
    if preset_choice == "Large":
        preset = int(preset_array/5)
    elif preset_choice == "Medium":
        preset = int(preset_array/8)
    elif preset_choice == "Small":
        preset = int(preset_array/16)
    
    model_choice = col2.radio(
    ":gear: - Please choose the Model: ",
    ["ConvNext", "RegNet", "Alexnet"],
    captions=[
        "***Recommended***",
        "",
        "***Baseline for Performance***"
    ])
    if model_choice == "ConvNext":
        model_path = 'models/convnext_small_v1_fixed.pt'  
    elif model_choice == "RegNet":
        model_path = 'models/regnet_y_16gf.pt'  
    elif model_choice == "Alexnet":
        model_path = "models/alexnet_13class_v1.pt"
    col3.image(img, caption="Uploaded Image", use_container_width=True)


    if st.button('Analyze', use_container_width=True):

        model = jit.load(model_path, map_location=device).eval()

        kwargs = dict(
            PYR_SCALE=1,
            WIN_STEP=preset,
            ROI_SIZE=(preset, preset),
            INPUT_SIZE=(preset, preset),
            VISUALIZE=True,
            MIN_CONF=0.2,
            VIZ_ROIS=False,
            SMOOTH_KERNEL = 3
        )

        detector = SWOD(model, device, **kwargs)

        preds = detector(img_array)
        saved_image_path = "./cache/map.jpg"
        
        col1,col2 = st.columns((1,1))
        col1.image(imgpath, caption="Original", use_container_width=True)
        col2.image(saved_image_path, caption="Mapped", use_container_width=True)

        with open("./cache/map.jpg", "rb") as file:
            btn = st.download_button(
                label="Download image",
                data=file,
                file_name="map.jpg",
                mime="image/jpg", use_container_width=True
            )
