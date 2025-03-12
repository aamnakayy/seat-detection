import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.title("Seat Detection Camera")

# Prefer back camera by setting constraints
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame  # Return frame unchanged

# Start camera with back-facing preference
webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=False
)