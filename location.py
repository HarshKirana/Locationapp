import streamlit as st
import base64
import googlemaps
import time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

gmaps = googlemaps.Client(key=maps_api_key)
df = pd.read_excel('Locality add.xlsx')

multipliers = {
    'hindu_temple': 1,
    'church': 2.5,
    'mosque': 5,
    'synagogue': 1,
    'buddhist_temple': 1,
    'sikh_gurdwara': 1
}

religious_affiliation = {
    'hindu_temple': 'Hindu',
    'church': 'Christian',
    'mosque': 'Muslim',
    'synagogue': 'Jewish',
    'buddhist_temple': 'Buddhist',
    'sikh_gurdwara': 'Sikh'
}

def search_places_of_worship(address, radius=500):
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return [], {}

    location = geocode_result[0]['geometry']['location']
    lat, lng = location['lat'], location['lng']
    query = "place of worship"

    results = gmaps.places_nearby(location=(lat, lng), radius=radius, keyword=query)
    places = []
    place_types = defaultdict(int)

    def process_results(results, places, place_types):
        for result in results['results']:
            places.append({
                'name': result.get('name'),
                'address': result.get('formatted_address'),
                'latitude': result['geometry']['location']['lat'],
                'longitude': result['geometry']['location']['lng']
            })
            for place_type in result.get('types', []):
                if place_type in ['church', 'mosque', 'synagogue', 'hindu_temple', 'buddhist_temple', 'sikh_gurdwara']:
                    place_types[place_type] += 1

    if results.get('results'):
        process_results(results, places, place_types)

        while 'next_page_token' in results:
            time.sleep(2)
            results = gmaps.places_nearby(page_token=results['next_page_token'])
            process_results(results, places, place_types)

    return places, place_types

def extract_lat_lng(data):
    return [(item['geometry']['location']['lat'], item['geometry']['location']['lng']) for item in data]

def collect_places(places, results, seen):
    for result in results['results']:
        identifier = (result.get('name'), result.get('vicinity'))
        if identifier not in seen:
            seen.add(identifier)
            places.append({
                'name': result.get('name'),
                'address': result.get('vicinity'),
                'type': ', '.join(result.get('types', []))
            })

def search_places(query):
    types = ['grocery_or_supermarket']
    places = []
    seen = set()

    for place_type in types:
        results = gmaps.places_nearby(location=query, radius=1000, type=place_type)
        collect_places(places, results, seen)
        while 'next_page_token' in results:
            time.sleep(2)
            results = gmaps.places_nearby(page_token=results['next_page_token'])
            collect_places(places, results, seen)

    return len(places)

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lon2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_nearest_mcdonalds(lat, lng):
    places_result = gmaps.places_nearby(location=(lat, lng), keyword='McDonald\'s', rank_by='distance')
    if places_result['results']:
        nearest_mcdonalds = places_result['results'][0]
        mcd_name = nearest_mcdonalds.get('name')
        mcd_namevicinity = nearest_mcdonalds.get('vicinity')
        res_name = f"The nearest McDonald's is {mcd_name} located at {mcd_namevicinity}."
    else:
        res_name = "No McDonald's found nearby."
    return res_name

def grid_search_places_of_worship(lat, lng, radius, step_size, grid_size):
    unique_places = set()
    base_lat, base_lng = lat, lng

    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            new_lat = base_lat + (i * step_size)
            new_lng = base_lng + (j * step_size)
            results = gmaps.places_nearby(location=(new_lat, new_lng), radius=radius, type=['neighborhood'])
            for result in results['results']:
                unique_places.add(result['name'])
    return unique_places
from PIL import Image
import io
def main():
    st.title("Retail Shop Location Analysis")
    address = st.text_input("Enter the address:", key="address")
    image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if image is not None:
        image1 = Image.open(image)
        st.image(image1, use_column_width=True)

    if st.button("Analyze") and address and image:
        radius = 1000
        image_path = f'uploadedimages/{image.name}'
        with open(image_path, "wb") as f:
            f.write(image.getbuffer())
        encoded_image = encode_image(image_path)

        geocode_result = gmaps.geocode(address)
        if not geocode_result:
            st.error('Address not found')
            return

        location = geocode_result[0]['geometry']['location']
        lat, lng = location['lat'], location['lng']

        places_count = search_places((lat, lng))

        locations = extract_lat_lng(geocode_result)
        for lat_lng in locations:
            df['Distance'] = df.apply(lambda row: haversine(lat_lng[1], lat_lng[0], row['latitude'], row['longitude']), axis=1)

        min_distance = df['Distance'].min()
        nearest_place = df[df['Distance'] <= min_distance]

        places, place_types = search_places_of_worship(address, radius)
        adjusted_totals = {ptype: count * multipliers.get(ptype, 1) for ptype, count in place_types.items()}
        total_adjusted = sum(adjusted_totals.values())
        estimated_population = int(nearest_place['pop_d'].values[0] * 2)
        demographics = {}
        for ptype, adjusted_count in adjusted_totals.items():
            percent = (adjusted_count / total_adjusted) * 100
            demographics[religious_affiliation[ptype]] = {
                'percent': percent,
                'estimated_population': (percent / 100) * estimated_population
            }

        majority_population = max(demographics.items(), key=lambda x: x[1]['estimated_population'])

        unique_places = list(grid_search_places_of_worship(lat, lng, radius=1000, step_size=0.005, grid_size=1))

        res_name = find_nearest_mcdonalds(lat, lng)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Is the place is right to start a retail shop here as the area information is given as the competition is {places_count} display count and  Nearest locality is {nearest_place['locality'].values[0]}, Approx Population: {nearest_place['pop_d'].values[0] * 5}, No. of Families: {(nearest_place['pop_d'].values[0] * 5) // 4} Tier of the city: {nearest_place['tier'].values[0]} also The majority population is {majority_population[0]} in 1km. So also suggest something for productivity according the majority religion."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 3000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        chat_gpt_response = response.json()['choices'][0]['message']['content']

        st.header("Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Places Count", value=places_count)
            st.metric(label="Tier of City", value=int(nearest_place['tier'].values[0]))
            st.metric(label="Population Estimate", value=int(nearest_place['pop_d'].values[0] * 5))
        with col2:
            st.metric(label="Family Estimate", value=int(nearest_place['pop_d'].values[0] * 5 // 4))
            st.metric(label="Majority Religion", value=majority_population[0])

        st.subheader("Nearby Places")
        for i in unique_places:
            st.write(i)

        st.subheader("Nearest McDonald's")
        st.write(res_name)

        st.subheader("ChatGPT Analysis")
        st.write(chat_gpt_response)

if __name__ == "__main__":
    main()
