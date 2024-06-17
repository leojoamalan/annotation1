import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import requests

@st.cache(allow_output_mutation=True)
def download_and_load_model(url, model_path="best_model.pth"):
    if not os.path.exists(model_path):
        # Download the model from a URL
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the download was successful
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success(f"Model downloaded successfully from {url}")
        except requests.RequestException as e:
            st.error(f"Failed to download the model: {e}")
            return None

    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
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

def predict_and_annotate(image, model, conf_threshold=0.35):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
    
    # Assume the output is for a classification task (modify as needed for other tasks)
    output = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(output, 1)
    
    if confidence.item() >= conf_threshold:
        result_text = f'Predicted class: {predicted_class.item()} with confidence {confidence.item():.2f}'
    else:
        result_text = f'No class met the confidence threshold of {conf_threshold}'
    
    # For segmentation, assuming binary mask example
    output = output.squeeze().cpu().numpy()
    mask = (output > conf_threshold).astype(np.uint8)
    mask_image = Image.fromarray(mask * 255)
    annotated_image = Image.blend(image.convert('RGBA'), mask_image.convert('RGBA'), alpha=0.5)
    
    return annotated_image, result_text

def main():
    st.title("Kidney Stone Annotation Tool")
    st.write("Upload an image to get started.")
    
    model_url = "https://your-model-repository-url/best_model.pth"
    model = download_and_load_model(model_url)
    
    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.write("Annotating...")
            annotated_image, result_text = predict_and_annotate(image, model)
            
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)
            st.write(result_text)

if __name__ == "__main__":
    main()
