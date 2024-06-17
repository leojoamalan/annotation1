import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import requests
import io

@st.cache(allow_output_mutation=True)
def load_model(model_url):
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        model = torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def main():
    st.title("Kidney Stone Annotation Tool")
    st.write("Upload an image to get started.")
    
    model_url ="https://github.com/leojoamalan/annotation/blob/main/best_model.pth"
    model = load_model(model_url)
    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
                
            st.write("Annotating...")
            annotated_image = model.predict(preprocess_image,conf=0.35)
                
            st.image(annotated_image.show(), caption='Annotated Image', use_column_width=True)

if __name__ == "__main__":
    main()
