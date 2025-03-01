import streamlit as st
from PIL import Image
import subprocess
from PIL import Image
import os

# Streamlit app title
# Fancy title with emojis, color, and larger font
st.markdown("""
    <h1 style='text-align: center; color: #10cc1d; font-size: 60px; font-family: "Monaco", monospace;
    '>Welcome to Maize Watch!</h1>
    <h3 style='text-align: center; color: #4CAF50;
    '>Please input Image and Other Information</h3>
""", unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Input Image Here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    input_path = "temp_input.png"
    output_path = "temp_output.png"
    image = Image.open(uploaded_file)
    image.save(input_path)

    result = subprocess.run(["python", "process_image_practice.py", input_path, output_path], capture_output=True, text=True)

    # Display the image
    st.image(image, caption="Uploaded Image", use_container_width=True)

   # Display the grayscale image
    if os.path.exists(output_path):
        grayscale_image = Image.open(output_path)
        st.image(grayscale_image, caption="Grayscale Image", use_container_width=True)

    # Display the output string
    output_text = result.stdout.strip()
    st.write(f"**{output_text}**")

    # Cleanup temp files
    os.remove(input_path)
    os.remove(output_path)