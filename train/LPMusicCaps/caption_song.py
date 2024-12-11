"""
NOTE: get_audio() and captioning() are from the lp-music-caps authors, everything else is ours
"""
import os
import argparse
from timeit import default_timer as timer
import torch
import numpy as np
import pandas as pd
from model.bart import BartCaptionModel
from utils.audio_utils import load_audio, STR_CH_FIRST

def pad_audio(audio, input_size):
    """
    Pads audio sequence to match input_size
    args:
        audio: audio sequence
        input_size: desired sequence length after padding
    """
    pad = np.zeros(input_size)
    pad[: audio.shape[-1]] = audio
    return pad

def get_audio(audio_path, duration=10, target_sr=16000, padding=False):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    # convert stereo to mono
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)
    input_size = int(n_samples)
    # pad sequence if too short
    if audio.shape[-1] < input_size:  
        audio = pad_audio(audio, input_size)
    # pad sequence to maximize chunk creation
    if padding and audio.shape[-1] % n_samples != 0:
        ceil = int(-(-audio.shape[-1] // n_samples))
        audio = pad_audio(audio, ceil * n_samples)
    else: # calc num chunks w/o padding
        ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

def captioning(args, model: torch.nn.Module, device: torch.device):
    audio_tensor = get_audio(audio_path = args.audio_path, padding=True)
    if device is not None:
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            text_prompt=args.question,
            num_beams=args.num_beams,
            use_nucleus_sampling=args.use_nucleus_sampling,
        )
    inference = ""
    number_of_chunks = range(audio_tensor.shape[0])
    for chunk, text in zip(number_of_chunks, output):
        time = f"[{chunk * 10}:00-{(chunk + 1) * 10}:00]"
        inference += f"{time}\n{text} \n \n"
    return inference

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Parse arguments for running LP-Music-Caps")
    
    # Add arguments
    parser.add_argument(
        "--audio_path", type=str, default="demo/samples/oh_mother_christina_aguilera.mp3", 
        help="Path to audio file (mp3 or wav)"
    )
    parser.add_argument(
        "--question", type=str, default="What instruments are heard in this song?", 
        help="Question to ask about the given song"
    )
    parser.add_argument(
        "--save_path", type=str, required=False, default="./captions.txt", 
        help="Path to where captions are saved"
    )
    parser.add_argument(
        "--pad_audio", action="store_true", default=True,
        help="Pad audio sequence if not perfectly divisible by target sample rate (target_sr)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="ckpt/mi_all_100.pth",
        help="checkpoint"
    )
    parser.add_argument(
        "--use_nucleus_sampling", action="store_true",
        help="To use nucleus sampling or not; if not, uses beam search by default. If beam search doesn't work, use this."
    )
    parser.add_argument(
        "--num_beams", type=int, default=10,
        help="To use nucleus sampling or not; if not, uses beam search"
    )
    
    # Parse arguments
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # create Bart model
    model = BartCaptionModel(max_length = 128)

    # load checkpoint
    pretrained_object = torch.load(args.checkpoint, map_location=device)
    state_dict = pretrained_object['state_dict']
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = model.cuda(device) # move model to device
    model.eval() # set model to inference mode

    description = captioning(
        args,
        model, 
        device,
    )

    # save captions to csv
    with open(args.save_path, "w+") as f:
        f.write(description)

if __name__ == "__main__":
    main()