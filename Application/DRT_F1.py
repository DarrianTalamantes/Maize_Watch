import streamlit as st
from PIL import Image

# Streamlit app title
# Fancy title with emojis, color, and larger font
st.markdown("""
    <h1 style='text-align: center; color: #10cc1d; font-size: 60px; font-family: "Monaco", monospace;
    '>Welcome to Maize Watch!</h1>
    <h3 style='text-align: center; color: #4CAF50;
    '>Please input Image and Other Information</h3>
""", unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
