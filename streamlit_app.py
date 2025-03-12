import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Seat Detection Camera")

# Start camera with back-facing preference and no audio
webrtc_streamer(
    key="camera",
    video_frame_callback=lambda frame: frame,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
)