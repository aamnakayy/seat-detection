import streamlit as st
import cv2
import numpy as np
import torch

st.title("Seat Detection")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

run = st.checkbox("Run Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera failed to initialize")
        break

    # ML detection
    results = model(frame)
    detections = results.pandas().xyxy[0]
    seats = detections[detections['name'].isin(['chair', 'bench'])]
    people = detections[detections['name'] == 'person']
    empty_seats = len(seats) - len(people) if len(seats) > len(people) else 0

    # Render results
    frame = results.render()[0]  # Draws boxes on frame
    FRAME_WINDOW.image(frame, channels="BGR")
    st.write(f"Empty Seats: {empty_seats}")
else:
    camera.release()