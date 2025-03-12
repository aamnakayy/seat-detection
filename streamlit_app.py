import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.title("Seat Detection Camera")

# Prefer back camera by setting constraints
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# Start camera with back-facing preference and no audio
webrtc_streamer(
    key="camera",
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=lambda frame: frame,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
)