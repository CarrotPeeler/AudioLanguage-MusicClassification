"""
Code modified and adapted from https://github.com/seungheondoh/lp-music-caps/blob/main/lpmc/music_captioning/preprocessor.py 
"""
import os
from datasets import load_dataset
from contextlib import contextmanager
import multiprocessing
import numpy as np
import json
from audio_utils import load_audio, STR_CH_FIRST
from tqdm import tqdm

# hard coding hparams
DATASET_PATH = "data/musiccaps"
MUSIC_SAMPLE_RATE = 16000
DURATION = 30 # whisper expected input length
DATA_LENGTH = int(MUSIC_SAMPLE_RATE * DURATION)
    
def msd_resampler(sample):
    path = sample
    save_name = os.path.join(DATASET_PATH,'npy', path.replace(".wav",".npy"))
    try:
        src, _ = load_audio(
            path=os.path.join(DATASET_PATH,'songs', path),
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
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))

def main():
    all_samples = os.listdir(DATASET_PATH + "/songs")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(msd_resampler, all_samples), total=len(all_samples)):
            pass
    print("finish extract")

if __name__ == '__main__':
    main()