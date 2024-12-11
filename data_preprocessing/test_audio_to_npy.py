"""
Code modified and adapted from https://github.com/seungheondoh/lp-music-caps/blob/main/lpmc/music_captioning/preprocessor.py 
"""
import os
import multiprocessing
import numpy as np
import csv
import sys
from audio_utils import load_audio, STR_CH_FIRST
from tqdm import tqdm

# hard coding hparams
DATASET_PATH = "ludwig"
MUSIC_SAMPLE_RATE = 16000
DURATION = 30 # whisper expected input length
DATA_LENGTH = int(MUSIC_SAMPLE_RATE * DURATION)

def get_all_audio_paths():
    # Directory where the mp3 files are stored
    root_dir = DATASET_PATH + "/mp3/mp3/"

    # This list will hold all the mp3 file paths
    mp3_paths = []

    # Walk through the directory and subdirectories
    for genre_dir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3'):  # Filter for mp3 files
                mp3_paths.append(os.path.join(genre_dir, file))
    return mp3_paths

def get_audio_paths(audio_csv):
    test_paths = []
    csv.field_size_limit(sys.maxsize)
    with open(audio_csv, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            test_paths.append(row['id'])
    all_paths = get_all_audio_paths()
    print(f"Found {len(all_paths)} audio clips")
    audio_paths = [
      path for path in all_paths 
      if path.rpartition('/')[-1].rpartition('.mp3')[0] in test_paths
    ]
    print(f"Loading {len(audio_paths)} audio clips")
    return audio_paths
    
def msd_resampler(sample):
    path = sample
    save_name = os.path.join(DATASET_PATH,'npy', path.rpartition('/')[-1].replace(".mp3",".npy"))
    try:
        src, _ = load_audio(
            path=path,
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
    except ValueError as err:
        print(f"{err} for {sample}")
        return
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.save(save_name, src.astype(np.float32))

def main():
    all_samples = get_audio_paths("datasets/split/test.csv")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(msd_resampler, all_samples), total=len(all_samples)):
            pass
    print("finish extract")

if __name__ == '__main__':
    main()