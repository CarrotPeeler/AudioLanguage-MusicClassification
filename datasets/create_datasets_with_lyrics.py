import os
import re
from langdetect import detect
import pandas as pd

# HARD-CODED PARAMS
LYRICS_START_TAG = "Lyrics"
LYRICS_END_TAG = "Embed"
EXTRACTED_LYRICS_PATH = (
    "extracted_lyrics_data"
)
TRACKS_CSV_PATH = "datasets/annotations/tracks.csv"

# Mapping of track_id to lyrics filename
hash_lyrics_files = None

def generate_hash_lyrics_files():
    global hash_lyrics_files

    if hash_lyrics_files is not None:
        return hash_lyrics_files

    hash_lyrics_files = {}

    for song_file in os.listdir(EXTRACTED_LYRICS_PATH):
        track_id = song_file.split("_")[0]
        hash_lyrics_files[track_id] = os.path.join(EXTRACTED_LYRICS_PATH, song_file)

    return hash_lyrics_files


def find_lyrics_file(track_id: str) -> str:
    """
    Searches through `extracted_lyrics_data` for the lyrics file (begins with the track id).

    :param track_id: The track id
    :return: The path to the lyrics
    """
    lyrics_files = generate_hash_lyrics_files()
    return lyrics_files.get(track_id, None)


def extract_clean_lyrics(lyrics: str) -> str:
    """
    Removes extraneous information from the lyrics

    :param lyrics: The lyrics
    :return: The cleaned lyrics
    """

    # Remove everything before the lyrics and after the lyrics
    start_lyrics = lyrics.index(LYRICS_START_TAG)
    end_lyrics = lyrics.index(LYRICS_END_TAG)

    lyrics = lyrics[start_lyrics + len(LYRICS_START_TAG) : end_lyrics]

    return lyrics


def get_lyrics(track_id: str) -> str:
    """
    Reads and extracts the lyrics from the lyrics file in `extracted_lyrics_data`

    :param track_id: The track id
    :return: The lyrics of the track
    """

    lyrics_file = find_lyrics_file(track_id)

    if lyrics_file is None:
        return None

    with open(lyrics_file, "r",  encoding="utf8") as f:
        lyrics = f.read()
        return extract_clean_lyrics(lyrics)


def get_or_unknown(data: dict, key: str) -> str:
    """
    Gets the value from the dictionary or returns "Unknown"

    :param data: The dictionary
    :param key: The key
    :return: The value or "Unknown"
    """

    return data.get(key, {}).get("S", "Unknown")


def get_clean_lyrics(track_id) -> dict:
    """
    Extracts the relevant fields from the track data

    :param track_data: The track data
    :return: A dictionary with the relevant fields
    """

    lyrics = get_lyrics(track_id)

    if lyrics is None:
        return None
    
    # Detect if lyrics are all English
    try:
        if detect(lyrics) != "en":
            print("Skipping track", track_id, "as lyrics are not in English")
            return None
    except:
        return None

    # Replace newlines with spaces
    lyrics = re.sub(r"\n", " ", lyrics)

    return lyrics


def main():
    df = pd.read_csv(TRACKS_CSV_PATH)

    # Add a new column 'lyrics' by fetching lyrics for each track
    df['lyrics'] = df['id'].apply(get_lyrics)

    # Save the updated DataFrame back to CSV
    df.to_csv('tracks_with_lyrics.csv', index=False)

    # Print first few rows to verify
    print(df.head())


if __name__ == "__main__":
    main()
