"""
Code adapted from https://github.com/keunwoochoi/audioset-downloader/blob/master/audioset_dl/__init__.py
"""
import csv
import datetime as dt
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
from yt_dlp import YoutubeDL
import subprocess
import argparse
from pathlib import Path

def _download_video_shell(x):
    (
        ytid,
        start,
        end,
        out_dir,
    ) = x
    start_dt, end_dt = dt.timedelta(milliseconds=start), dt.timedelta(milliseconds=end)
    ydl_opts = {
        "outtmpl": f"{out_dir}/[{ytid}]-[{start // 1000}-{end // 1000}].%(ext)s",
        "format": "(bestvideo[height<=640]/bestvideo[ext=webm]/best)+(bestaudio[ext=webm]/best[height<=640])",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",  # one of avi, flv, mkv, mp4, ogg, webm
            }
        ],
        "quiet": True,
        "no-mtime": True,
    }
    yturl = f"https://youtube.com/watch?v={ytid}"
    section_opt = f"*{start_dt}-{end_dt}"
    cmd = (
        f'yt-dlp -f "{ydl_opts["format"]}" {yturl} '
        f"--download-sections {section_opt} "
        f"--quiet "
        f'--output "{ydl_opts["outtmpl"]}"'
    )
    try:
        # time.sleep(0.1)
        subprocess.run(cmd, shell=True, timeout=100)
    except subprocess.CalledProcessError as e:
        print(e)
    except KeyboardInterrupt:
        raise

def _download_audio(x):
    (
        ytid,
        start,
        end,
        out_dir,
    ) = x
    # print(start, end)
    start_dt, end_dt = dt.timedelta(milliseconds=start), dt.timedelta(milliseconds=end)
    ydl_opts = {
        "outtmpl": f"{out_dir}/[{ytid}]-[{start//1000}-{end//1000}].%(ext)s",
        "format": "bestaudio[ext=webm]/bestaudio/best",
        "external_downloader": "ffmpeg",
        "external_downloader_args": [
            "-ss",
            str(start_dt),
            "-to",
            str(end_dt),
            "-loglevel",
            "panic",
            "-http_proxy",
            "socks5://127.0.0.1:1080"
        ],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "no-mtime": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
    except KeyboardInterrupt:
        raise
    except Exception:
        pass

def download_ps(ytid, st_list, ed_list, save_path, target, num_processes=None, desc=None):
    with mp.Pool(processes=num_processes) as pool, tqdm(total=len(ytid), desc=desc) as pbar:
        if target == "audio":
            for _ in tqdm(
                pool.imap(
                    _download_audio,
                    zip(ytid, st_list, ed_list, [save_path] * len(ytid)),
                ),
                total=len(ytid),
            ):
                pbar.update()
        elif target == "video":
            for _ in tqdm(
                pool.imap(
                    _download_video_shell,
                    zip(ytid, st_list, ed_list, [save_path] * len(ytid)),
                )
            ):
                pbar.update()
        else:
            raise NotImplementedError(f"target {target} is not implemented yet.")

def dl_audioset(args):
    youtube_ids = []
    start_times = []
    end_times = []

    # Open and read the CSV file
    with open(args.csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)  
        for row in reader:
            # Append each column's value to the respective list
            youtube_ids.append(row['ytid'])
            start_times.append(row['start_s'])
            end_times.append(row['end_s'])

    start_time = [int(st)*1000 for st in start_times]
    end_time =  [int(et)*1000 for et in end_times]

    download_ps(youtube_ids, start_time, end_time, args.save_path, args.target, num_processes=max(1, max(1, mp.cpu_count()-10)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data_preprocessing/MusicCaps/MusicCaps.csv", type=str)
    parser.add_argument("--save_path", default="data/musiccaps/songs", type=str)
    parser.add_argument("--target", default="audio", choices=['audio', 'video'], type=str)
    args = parser.parse_args()
    dl_audioset(args=args)