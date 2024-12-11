import os
import csv
import json

"""
FILTER BASED ON AVAILABLE MP3s
"""

# hard coding hparams
DATASET_PATH = "/content/ludwig"

def get_all_audio_paths():
    # Directory where the mp3 files are stored
    root_dir = DATASET_PATH + "/mp3/mp3/"

    # This list will hold all the mp3 file paths
    mp3_paths = []

    # Walk through the directory and subdirectories
    for genre_dir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3'):  # Filter for mp3 files
                mp3_paths.append(file.replace(".mp3", ""))
    return mp3_paths

all_fnames = get_all_audio_paths()

print(f"Found {len(all_fnames)} audio clips")

# Load the JSON data from a file
with open(f'{DATASET_PATH}/labels.json', 'r') as file:
    json_data = json.load(file)

# Prepare the data to be written to CSV
csv_data = []
missing_ids = 0
# Loop through each track and extract necessary fields
for track_id, track_info in json_data['tracks'].items():
    if track_id not in all_fnames:
        missing_ids += 1
        continue
    try:
        # Extract the required fields (access the nested dictionaries with ['S'])
        artist = track_info['artist']['S']
        name = track_info['name']['S']
        album = track_info['album']['S']
        track_type = track_info['type']['S']
        genre = track_info['genre']['S']
        genre_dir = genre if genre != "funk / soul" else "funk _ soul"
        mp3_path = os.path.join("ludwig", "mp3", "mp3", genre_dir, track_id)
        # Convert the subgenres list into a string (comma-separated)
        subgenres = ', '.join([subgenre['S'] for subgenre in track_info['subgenres']['L']])
        
        # Append to the CSV data list
        csv_data.append([track_id, artist, name, album, track_type, genre, subgenres])
    
    except KeyError as e:
        missing_ids += 1

print(f"Number of missing tracks: {missing_ids}")

# Define CSV file name
csv_filename = "tracks.csv"

# Write the CSV data to a file
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['id', 'artist', 'name', 'album', 'type', 'genre', 'subgenres'])
    # Write the data rows
    writer.writerows(csv_data)

print(f"CSV file '{csv_filename}' has been created.")
