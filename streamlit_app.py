import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Seat Detection Camera")

# Start the camera immediately
webrtc_streamer(key="camera", video_frame_callback=lambda frame: frame)