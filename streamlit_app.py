import streamlit as st
import torch
from PIL import Image
import io

# Load pre-trained YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model

model = load_model()

# Title of the app
st.title("Object Detection App with YOLOv5")

# Instructions
st.write("Use your webcam to take a picture, and the model will detect objects.")

# Camera input widget
picture = st.camera_input("Take a picture")

# Process the image with YOLOv5
if picture is not None:
    # Display the captured image
    st.image(picture, caption="Captured Image", use_column_width=True)  # Changed here

    # Convert Streamlit's BytesIO to PIL Image
    img = Image.open(picture)

    # Run inference
    results = model(img)

    # Extract and display predictions
    detections = results.pandas().xyxy[0]
    if not detections.empty:
        st.write("Detected objects:")
        for _, row in detections.iterrows():
            class_name = row["name"]
            confidence = row["confidence"]
            st.write(f"- {class_name} (Confidence: {confidence:.2f})")
    else:
        st.write("No objects detected in the image.")

    # Display the image with bounding boxes
    st.image(results.render()[0], caption="Image with Detections", use_column_width=True)  # Changed here