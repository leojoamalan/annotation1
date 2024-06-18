
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
    label = 'Kidney stone'  # Get the class label
    confidence = detection['confidence']  # Get the confidence level

    # Define colors
    box_color = "blue"  # Color of the bounding box
    text_color = "white"  # Color of the text

    # Draw the bounding box
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    draw.rectangle([left, top, right, bottom], outline=box_color, width=2)

    # Draw the points
    for point in points:
        px, py = point['x'], point['y']
        draw.ellipse((px-1, py-1, px+1, py+1), fill="blue")

    # Add label and confidence level on top of the bounding box
    font = ImageFont.load_default()  # You can change the font and size here
    draw.text((left, top - 20), f"{label} ", fill=text_color, font=font)

def draw_not_found(draw, image_size):
    font = ImageFont.load_default()
    text = "Kidney Stone Not Found"
    text_width, text_height = 225,300
    text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(text_position, text, fill="blue", font=font)

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

    # Check if there are detections
    if 'predictions' in result and len(result['predictions']) > 0:
        # Draw the detections on the input image
        for detection in result['predictions']:
            draw_detection(draw, detection)
    else:
        # If no detections found, draw a message on the image
        draw_not_found(draw, image.size)

    # Convert image to byte array for displaying
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
if st.button('Annotate'):
    # Display the result image with detections or not found message
    st.image(image, caption='Processed Image.', use_column_width=True)

