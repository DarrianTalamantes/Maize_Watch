import requests
import csv

# A dictionary to map state FIPS codes to state names
state_fips_to_name = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California", 
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia", 
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois", 
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana", 
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", 
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada", 
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York", 
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", 
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota", 
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", 
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
}

# Load GeoJSON file for USA counties from an alternative source
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(url)

if response.status_code == 200:
    counties_geojson = response.json()

    # Open a CSV file to write data
    with open('counties_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["County Code", "State Name", "County Name"])

        # Iterate through the counties and extract the necessary information
        for county in counties_geojson['features']:
            county_code = county['properties']['COUNTY']
            state_code = county['properties']['STATE']
            county_name = county['properties']['NAME']
            
            # Map state FIPS code to state name
            state_name = state_fips_to_name.get(state_code, "Unknown State")
            
            # Write the row with county code, state name, and county name
            writer.writerow([county_code, state_name, county_name])

    print("CSV file 'counties_data.csv' has been created.")
else:
    print(f"Failed to retrieve data: {response.status_code}")
