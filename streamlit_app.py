# # import streamlit as st
# # import torch
# # from torchvision import transforms
# # from PIL import Image, ImageDraw
# # import numpy as np
# # import os
# # import requests
# # import io

# # @st.cache(allow_output_mutation=True)
# # # def load_model(model_url):
# # #     try:
# # #         response = requests.get(model_url)
# # #         response.raise_for_status()
# # #         model = torch.load(response.content, map_location=torch.device('cpu'))
# # #         model.eval()
# # #         return model
# # #     except Exception as e:
# # #         st.error(f"Failed to load model: {e}")
# # #         return None

# # def preprocess_image(image):
# #     preprocess = transforms.Compose([
# #     transforms.Resize(256),
# #     transforms.CenterCrop(224),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])
# #     img_tensor = preprocess(image)
# #     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
# #     return img_tensor

# # def main():
# #     st.title("Kidney Stone Annotation Tool")
# #     st.write("Upload an image to get started.")
    
# #     model_url ="https://drive.google.com/uc?export=download&id=1MPkwGD6Jx0b63mvdCKdIBubtCDHMaZry"
# #     # model = load_model(model_url)
# #     MODEL_PATH = "best_model.pth"

# #     if not os.path.exists(MODEL_PATH):
# #         st.info("Downloading model...")
# #         response = requests.get(model_url)
# #         with open(MODEL_PATH, 'wb') as f:
# #             f.write(response.content)
# #         st.success("Model downloaded!") 
# #     # Load your pre-trained model
# #     model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# #     model.eval()
# #     if model:
# #         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# #         if uploaded_file is not None:
# #             image = Image.open(uploaded_file)
# #             st.image(image, caption='Uploaded Image', use_column_width=True)
                
# #             st.write("Annotating...")
# #             annotated_image = model.predict(preprocess_image(image),conf=0.35)
                
# #             st.image(annotated_image.show(), caption='Annotated Image', use_column_width=True)

# # if __name__ == "__main__":
# #     main()



# import streamlit as st
# from PIL import Image, ImageDraw,ImageFont
# import io

# # Dummy detection result for demonstration purposes
# import inference_sdk
# from inference_sdk import InferenceHTTPClient

# # initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="kAbNHf7BK7n4dF8MzsMh"
# )

# # infer on a local image
# #result = CLIENT.infer(image, model_id="kidney-22s5u/1")

# # Function to draw bounding boxes and points
# def draw_detection(draw, detection):
#     x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
#     points = detection['points']

#     # Draw the bounding box
#     left = x - w / 2
#     top = y - h / 2
#     right = x + w / 2
#     bottom = y + h / 2
#     draw.rectangle([left, top, right, bottom], outline="red", width=2)

#     # Draw the points
#     for point in points:
#         px, py = point['x'], point['y']
#         draw.ellipse((px-1, py-1, px+1, py+1), fill="blue")
# def annotate_image(image, text):
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.load_default()  # You can change the font and size here
#     draw.text((10, 10), text, fill="black", font=font)

# st.title("Kidney Stone Annotation")

# # File uploader for image upload
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     # Load the uploaded image
#         image = Image.open(uploaded_file)
#         draw = ImageDraw.Draw(image)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#         result = CLIENT.infer(image, model_id="kidney-22s5u/1")
#         # Draw the detections on the input image
#         for detection in result['predictions']:
#             draw_detection(draw, detection)
#         annotate_image(image, "Detected Objects")
#         # Convert image to byte array for displaying
#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='PNG')
#         img_byte_arr = img_byte_arr.getvalue()
# if st.button('Annotate'):
#         # Display the result image with detections
#         st.image(image, caption='Processed Image.', use_column_width=True)
        
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from inference_sdk import InferenceHTTPClient

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="kAbNHf7BK7n4dF8MzsMh"
)

# Function to draw bounding boxes and points with labels
def draw_detection(draw, detection):
    x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
    points = detection['points']
    label = 'Kidney Stone ' # Get the class label
    confidence = detection['confidence']  # Get the confidence level

    # Draw the bounding box
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    draw.rectangle([left, top, right, bottom], outline="blue", width=2)

    # Draw the points
    for point in points:
        px, py = point['x'], point['y']
        draw.ellipse((px-1, py-1, px+1, py+1), fill="blue")

    # Add label and confidence level on top of the bounding box
    font = ImageFont.load_default()  # You can change the font and size here
    draw.text((left, top - 20), f"{label}", fill="white", font=font)

st.title("Kidney Stone Annotation")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    draw = ImageDraw.Draw(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform inference using your inference client
    result = CLIENT.infer(image, model_id="kidney-22s5u/1")

    # Draw the detections on the input image
    for detection in result['predictions']:
        draw_detection(draw, detection)

    # Convert image to byte array for displaying
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Display the result image with detections
    st.image(image, caption='Processed Image.', use_column_width=True)
