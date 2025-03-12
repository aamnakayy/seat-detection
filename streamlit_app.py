import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import torch
import numpy as np

st.title("Seat Detection Camera")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def video_frame_callback(frame):
    # Convert frame to numpy array (BGR format from WebRTC)
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv5 inference
    results = model(img)
    detections = results.pandas().xyxy[0]  # Get detections as a pandas DataFrame
    
    # Filter for chairs and people
    chairs = detections[detections['name'].isin(['chair'])]  # YOLOv5 labels 'chair'
    people = detections[detections['name'] == 'person']
    
    # Calculate empty chairs (assuming chairs without people nearby are empty)
    total_chairs = len(chairs)
    total_people = len(people)
    empty_chairs = max(total_chairs - total_people, 0)  # No negative count
    
    # Draw detections on the frame
    img_with_detections = results.render()[0]  # Rendered frame with boxes
    
    # Display empty chair count
    st.write(f"Empty Chairs: {empty_chairs}")
    
    # Convert back to WebRTC frame (expects RGB, but render gives BGR)
    return cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

# Start camera with back-facing preference and no audio
webrtc_streamer(
    key="camera",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
)