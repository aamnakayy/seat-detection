import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import torch

st.title("Seat Detection")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img)
    detections = results.pandas().xyxy[0]
    seats = detections[detections['name'].isin(['chair', 'bench'])]
    people = detections[detections['name'] == 'person']
    empty_seats = len(seats) - len(people) if len(seats) > len(people) else 0
    frame = results.render()[0]
    st.write(f"Empty Seats: {empty_seats}")
    return frame

webrtc_streamer(key="seat_detection", video_frame_callback=video_frame_callback)