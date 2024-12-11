# Multi-Modal Subgenre Classification with Lyrics and Audio Captions

## Dependencies
### Option 1: Run locally
1. Python 3.8+
2. Linux environment and Cuda GPU
3. FFmpeg
```
sudo apt install ffmpeg
```
4. Pip Packages:
```
pip install -r requirements.txt
``` 
### Option 2: Use our notebook

## Training the 1st Stage (Audio-to-Text Music Captioning)
### Download and Preprocess MusicCaps Audio
You can download the Mel-Spectrogram .npy files from [here](https://wpi0-my.sharepoint.com/:u:/g/personal/jchan3_wpi_edu/EfNwt3eIqf1CtedKM6AEgDIBS-rw3h1e8MFw4MhhyLpUxQ?e=En2qxt)

Manual audio file download and conversion:
```
python data_preprocessing/download_youtube_audio.py
```

Now, preprocess the wav files into npy:
```
python data_preprocessing/audio_to_npy.py
```

### Fine-tuning LP-Music-Caps BART for Audio Question-Answering
We adjust the LP-Music-Caps BART encoder to enable instruction prompting. This allows us to guide the generation of our music captions.

There are four model configurations you can train:
- MusicInstruct-short: model only gives short answers for simple, close-ended questions
- MusicInstruct-long: moded only gives long answers for more subjective, open-ended questions
- MusicInstruct-short-long: model pretrained for giving short answers; fine-tuned for giving longer answers to more open-ended questions
- MusicInstruct-all: model fine tuned for giving both short and long answers, uses judgment to determine if questions require longer or shorter answers

Edit the config file you're using with correct file paths and hyperparams, then run something similar:
```
python3 train/LPMusicCaps/transfer.py --yaml_config_path=train/LPMusicCaps/exp/transfer/music_instruct/mi_short_long_hparams.yaml
```

### Generating Captions from Audio Question-Answering
```
python3 data_preprocessing/test_audio_to_npy.py
```
```
python3 train/LPMusicCaps/captions_from_qa.py
```
Make sure to apply the checkpoint you want to use with `--checkpoint`. Inference is quite slow because of `generate`.

### Checkpoints for Fine-tuned LP-Music-Caps BART Model Weights
- [BART-MusicInstruct-Short](https://wpi0-my.sharepoint.com/:u:/g/personal/jchan3_wpi_edu/EbmxtFORqpBOqMbUTuH9r8UBRVpRkm2jTQpdpYAiD5hvtA?e=HT4fZ0)
- [BART-MusicInstruct-Short-Long](https://wpi0-my.sharepoint.com/:u:/g/personal/jchan3_wpi_edu/EdrTtru-udFIvIRYU5F7VAYBY7PAbmnMWrPsSoqpDH_UBw?e=LoGiGH)
- [BART-MusicInstruct-All](https://wpi0-my.sharepoint.com/:u:/g/personal/jchan3_wpi_edu/Ea4Y9ZKd-oxGv4csCqiq5c8B9W3vsbnwgj1hoXoypyIpUQ?e=gzj0b2)


## Acknowledgements

We utilized the following resources for developing multimodal training and code:

- [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) for our 1st stage audio model checkpoint
- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) for our 1st stage llm model checkpoint
- [AnyModal](https://github.com/ritabratamaiti/AnyModal) for training multimodal models
- [MusiLingo](https://github.com/zihaod/MusiLingo) for implementation reference and the MusicInstruct dataset
- [LP-Music-Caps](https://github.com/seungheondoh/lp-music-caps) for utility code


