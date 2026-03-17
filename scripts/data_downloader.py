import os
import urllib.request
import re

# Base URL for the directory
base_url = 'https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/'

# Create the target folder if it doesn't exist
os.makedirs('../data/raw/chirps', exist_ok=True)

# Fetch the HTML content of the directory
with urllib.request.urlopen(base_url) as response:
    html = response.read().decode('utf-8')

# Find all .tif file links using regex (matching chirps-v3.0.YYYY.MM.tif)
tif_links = re.findall(r'href="(chirps-v3.0\.\d{4}\.\d{2}\.tif)"', html)

# Download each file
for filename in tif_links:
    file_url = base_url + filename
    save_path = os.path.join('../data/raw/chirps', filename)
    print(f"Downloading {filename} to {save_path}...")
    urllib.request.urlretrieve(file_url, save_path)
    print(f"Downloaded {filename}")

print("All files downloaded successfully!")