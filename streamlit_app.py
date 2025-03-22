import streamlit as st

# Title of the app
st.title("Seat Detection App")

# Instructions
st.write("Use your webcam to take a picture of a seat.")

# Camera input widget
picture = st.camera_input("Take a picture")

# Display the captured image (if any)
if picture is not None:
    st.image(picture, caption="Captured Image", use_column_width=True)
    st.write("Image captured successfully! You can now process it with your model.")