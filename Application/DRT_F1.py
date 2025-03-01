import streamlit as st
from PIL import Image
import subprocess
import os
import plotly.express as px
import requests
import plotly
import pandas as pd


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
    st.markdown("## Original Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    st.markdown("----------------------")

    # Display the output string
    output_text = result.stdout.strip()
    if output_text != "Healthy":
        st.markdown("## Corn appears to be infected with")
        st.markdown(f"## **{output_text}**")    
        st.markdown("----------------------")
        st.markdown("## Predicted Desiese Hot Spots")
        # Display the grayscale image

        if os.path.exists(output_path):
            grayscale_image = Image.open(output_path)
            st.image(grayscale_image, caption="Grayscale Image", use_container_width=True)

    else:
        st.markdown("### Congratulations, your corn appears")
        st.markdown(f"### **{output_text}**")
        st.markdown("### 🌽 😄 🌽")


    ########## This displays a USA map with all counties and desieses #################

st.markdown("----------------------")

# Set the Mapbox access token (use your own Mapbox token here)
plotly.express.set_mapbox_access_token("your-mapbox-access-token")

# Load GeoJSON file for USA counties from an alternative source

url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(url)

if response.status_code == 200:
    counties_geojson = response.json()

    # Load the CSV data containing county data
    df = pd.read_csv('/DataBase/counties_data.csv')

    # Calculate the sum of columns 3 and beyond for each row
    df['sum_columns'] = df.iloc[:, 2:].sum(axis=1)

    # Create a dictionary of county FIPS codes to sum of values
    county_fips_to_sum = dict(zip(df['FIPS'], df['sum_columns']))

    # Create a new color list based on the sum of values
    color_values = [county_fips_to_sum.get(county['id'], 0) for county in counties_geojson['features']]

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        geojson=counties_geojson,
        locations=[county['id'] for county in counties_geojson['features']],  # Use FIPS code as location
        color=color_values,  # Color based on the sum of columns 3 and beyond
        hover_name=[county['properties']['NAME'] for county in counties_geojson['features']],  # County name on hover
        color_continuous_scale="Viridis",  # Choose a color scale
        title="USA Counties Map",
        mapbox_style="open-street-map",  # Map style
        height=600,
        width=800,
        center={"lat": 37.0902, "lon": -95.7129},  # Latitude and longitude of the USA
        zoom=3  # Set zoom level to focus on the USA
    )

    # Display the map in Streamlit
    st.plotly_chart(fig)

else:
    print(f"Failed to retrieve data: {response.status_code}")


    # Cleanup temp files
    os.remove(input_path)
    os.remove(output_path)