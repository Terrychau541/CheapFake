##Clone the repo and open set up the model
import streamlit as st
import pandas as pd
import os

'''
# Facial Editing Detector
'''
from subprocess import call
import global_classifier_modified3 as gcm

image = st.file_uploader("Select an image", type = ["png", "jpg"])

if image is not None:
    st.image(image)
    prob = gcm.classify_fake(image)

    st.write("The probability of facial editing is {:.2f}%".format(prob*100))



#python global_classifier.py edited.jpg --model_path weights/global.pth
