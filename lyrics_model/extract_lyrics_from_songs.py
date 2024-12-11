import os
import re
import time
from requests.exceptions import RequestException
import lyricsgenius as lg
import json

# load JSON file
with open('Ludwig Dataset/labels.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

song_info_list = []


# create list of song info from JSON data
for track_id, track_info in data["tracks"].items():
    name = track_info.get("name", {}).get("S", "Unknown")
    artist = track_info.get("artist", {}).get("S", "Unknown")
    song_info = [track_id, name, artist]
    song_info_list.append(song_info)

print(len(song_info_list))


api_key = "boluUqxbfjZBST_com8huaD8NZu41mpvMctlQ440roHLGgDnOfdXaMl1IHoYvZ2j"

lg.api.TIMEOUT = 120
genius = lg.Genius(api_key, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)

folder_name = "extracted_lyrics_data"


def fetch_lyrics(song_info_list, folder_name, no_lyrics_folder="no_lyrics"):
    # make sure the folder exists
    os.makedirs(folder_name, exist_ok=True)
    
    # keep track of songs to process
    remaining_songs = song_info_list[:]

    while remaining_songs:
        for song_info in remaining_songs[:]:  # copy the list to modify while iterating
            try:
                id = song_info[0]
                title = song_info[1]
                artist = song_info[2]

                # clean filenames
                sanitized_title = re.sub(r'[\\/*?:"<>|]', "", title)
                sanitized_artist = re.sub(r'[\\/*?:"<>|]', "", artist)
                filename = f"{id}_{sanitized_title}_{sanitized_artist}_lyrics.txt"
                filepath = os.path.join(folder_name, filename)

                # skip if file already exists
                if os.path.exists(filepath):
                    print(f"File {filename} already exists. Skipping...")
                    remaining_songs.remove(song_info)
                    continue

                # if song has no lyrics (previously determined)
                no_lyrics_path = os.path.join(no_lyrics_folder, filename)
                if os.path.exists(no_lyrics_path):
                    print(f"File {filename} has no lyrics. Skipping...")
                    remaining_songs.remove(song_info)
                    continue

                # search for lyrics
                song = genius.search_song(title, artist)
                time.sleep(3)  # avoid hitting rate limits

                # check if there are lyrics in the song
                if len(filepath) > 128:
                    print(f"Filename too long: {filename}. Skipping...")
                    remaining_songs.remove(song_info)
                    continue
                elif song is None:
                    print(f"Lyrics not found for: {title} by {artist}")
                    # write to no_lyrics folder for future reference
                    with open(no_lyrics_path, "w", encoding="utf-8") as file:
                        file.write("No lyrics found")
                else:
                    # write to folder
                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(song.lyrics)
                    print(f"Lyrics saved to {filename}")
                # remove from the list once processed
                remaining_songs.remove(song_info)
            # if timeout request, retry after a short delay
            except RequestException as e:
                print(f"Request error: {e}. Retrying after a short delay...")
                time.sleep(10) 

            # exit the loop, leaving `remaining_songs` intact
            except KeyboardInterrupt:
                print("Interrupted by user. Resuming...")
                break  
        
        # additional wait before retrying all remaining songs
        if remaining_songs:
            print(f"Retrying {len(remaining_songs)} remaining songs...")
            time.sleep(30)  


fetch_lyrics(song_info_list, folder_name)
print("Lyrics extraction complete.")
