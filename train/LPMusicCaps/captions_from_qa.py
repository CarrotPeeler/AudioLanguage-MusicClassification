import os
import sys
import csv
import argparse
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
import random
from model.bart import BartCaptionModel

def load_audio(npy_path, sr=16000, duration=10):
    n_samples = int(sr * duration)
    audio = np.load(npy_path, mmap_mode='r')

    if len(audio.shape) == 2:
        audio = audio.squeeze(0)
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:
        pad = np.zeros(input_size)
        pad[:audio.shape[-1]] = audio
        audio = pad
    audio_tensor = torch.from_numpy(np.array(audio[n_samples:2*n_samples]).astype('float32'))
    return audio_tensor.unsqueeze(0)

def captioning(args, model: torch.nn.Module, device: torch.device):
    # Initialize result variable
    final_results = []

    # Loop through audio files
    for audio_path in tqdm(os.listdir(args.npy_dir)):
        audio_tensor = load_audio(os.path.join(args.npy_dir, audio_path))
        
        if device is not None:
            audio_tensor = audio_tensor.to(device)
        
        # Initialize caption paragraph for this audio
        paragraph_caption = {
            "[0:00-10:00]": [],
            "[10:00-20:00]": [],
            "[20:00-30:00]": [],
        }

        # Precision context based on args.mixed_precision
        if args.mixed_precision == "fp16":
            autocast_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None  # No mixed precision

        # Generate captions for each question
        for question in args.questions:
            with torch.no_grad():
                if autocast_dtype:
                    with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                        output = model.generate(
                            samples=audio_tensor,
                            text_prompt=question,
                            num_beams=args.num_beams,
                            use_nucleus_sampling=args.use_nucleus_sampling,
                        )
                else:
                    output = model.generate(
                        samples=audio_tensor,
                        text_prompt=question,
                        num_beams=args.num_beams,
                        use_nucleus_sampling=args.use_nucleus_sampling,
                    )

            # Collect output and concatenate responses for this question
            for i, temporal_answer in enumerate(output):
                timestamp = f"[{i * 10}:00-{(i + 1) * 10}:00]"
                paragraph_caption[timestamp].append(temporal_answer.strip())
        
        # Combine responses into a single descriptive paragraph for the audio
        paragraph_caption = " ".join(f"{timestamp} {' '.join(contents)}" for timestamp, contents in paragraph_caption.items())
        final_results.append({"audio_path": audio_path, "caption": paragraph_caption})

    # Save captions to CSV
    with open(args.save_path, "w+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "caption"])
        writer.writeheader()
        writer.writerows(final_results)

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Parse arguments for running LP-Music-Caps")
    
    # Add arguments
    parser.add_argument(
        "--npy_dir", type=str, default="ludwig/npy", 
        help="Path to csv with audio file paths (mp3 or wav)"
    )
    parser.add_argument(
        "--save_path", type=str, required=False, default="./captions.txt", 
        help="Path to where captions are saved"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/lp-music-caps_mi-short/mi_short_100eps.pth",
        help="checkpoint"
    )
    parser.add_argument(
        '--questions', nargs='+', default=[
            "What instruments are heard in the song and are they acoustic or electronic?",
            "What is the rhythm and tempo of the song?",
            "What is the mood and theme of the song?",
            "How does the song progress?",
            "Are there distinct cultural or stylistic influences present in the song?"
        ],
        help='List of questions to ask for guiding caption generation')
    
    parser.add_argument(
        "--use_nucleus_sampling", action="store_true",
        help="To use nucleus sampling or not; if not, uses beam search by default. If beam search doesn't work, use this."
    )
    parser.add_argument(
        "--num_beams", type=int, default=10,
        help="To use nucleus sampling or not; if not, uses beam search"
    )
    parser.add_argument(
        "--mixed_precision", type=str, choices=["fp16", "bf16", "none"], default="none",
        help="Choose mixed precision mode: fp16, bf16, or none"
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

    captions = captioning(
        args,
        model, 
        device,
    )

if __name__ == "__main__":
    main()