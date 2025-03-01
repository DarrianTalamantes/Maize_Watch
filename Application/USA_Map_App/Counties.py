import plotly.express as px
import requests
import plotly

# Set the Mapbox access token (use your own Mapbox token here)
plotly.express.set_mapbox_access_token("your-mapbox-access-token")

# Load GeoJSON file for USA counties from an alternative source
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(url)

if response.status_code == 200:
    counties_geojson = response.json()

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        geojson=counties_geojson,
        locations=[county['id'] for county in counties_geojson['features']],  # Use FIPS code as location
        color=[1]*len(counties_geojson['features']),  # Color for all counties (replace with actual data)
        hover_name=[county['properties']['NAME'] for county in counties_geojson['features']],  # County name on hover
        color_continuous_scale="Viridis",  # Choose a color scale
        title="USA Counties Map",
        mapbox_style="open-street-map",  # Map style
        height=600,
        width=800,
        # Center the map on the USA and zoom in
        center={"lat": 37.0902, "lon": -95.7129},  # Latitude and longitude of the USA
        zoom=3  # Set zoom level to focus on the USA
    )

    # Show the map
    # Save the map to an HTML file
    fig.write_html("us_counties_map.html")
    print("Map saved as us_counties_map.html")

else:
    print(f"Failed to retrieve data: {response.status_code}")
