"""
Code for loading the MusicCaps dataset (https://huggingface.co/datasets/google/MusicCaps)
"""
from utils.base_dataset import BaseDataset
import os
import torch
import numpy as np
import random
import torchaudio.transforms as T
from datasets import load_dataset

class LPMCDataset(BaseDataset):
    def __init__(
            self, 
            processor, 
            data_dir, 
            ann_path, 
            split, 
            val_subset_size=None,
            inflate=False,
        ):
        super().__init__()
        self.split = split # train or test
        self.data_dir = data_dir #music_data
        self.resample_rate = processor.sampling_rate
        self.processor = processor

        self.all_ann = load_dataset("seungheondoh/LP-MusicCaps-MC")
        self.all_ann, self.npy_fnames = self.filter_json_by_npy(self.all_ann, self.data_dir)
        
        self.ann = []
        if self.split == "all":
            # Include all annotations
            self.ann = self.all_ann
        elif self.split == "train":
            # Use training dataset
            self.ann = [row for row in self.all_ann if row['is_crawled'] and row['split'] == "train"] 
        elif self.split == "test":
            # Use test dataset
            self.ann = [row for row in self.all_ann if row['is_crawled'] and row['split'] == "test"] 
        elif self.split == "val":
            # Create a validation subset from the training dataset
            val_items = [row for row in self.all_ann if row['is_crawled'] and row['split'] == "test"] 
            if val_subset_size is not None and val_subset_size < len(val_items):
                self.ann = random.sample(val_items, val_subset_size)
            else:
                self.ann = val_items

        # inflate dataset size by reusing songs with different captions
        if inflate:
            self.inflate()


    def inflate(self):
        # keys for the different caption versions
        caption_keys = [
            'caption_ground_truth',
            'caption_writing',
            'caption_summary',
            'caption_paraphrase',
            'caption_attribute_prediction'
        ]

        # List to store the expanded dataset
        expanded_ann = []

        # Duplicate each sample 5 times with a different caption version
        for sample in self.ann:
            for key in caption_keys:
                # Create a copy of the sample
                duplicated_sample = sample.copy()
                # Set the 'caption' key to the current caption version
                duplicated_sample['caption'] = sample[key]
                # Add the duplicated sample to the expanded list
                expanded_ann.append(duplicated_sample)

        # Replace the original self.ann with the expanded dataset
        self.ann = expanded_ann

    def get_max_length(self):
        return "N/A", "N/A"

    def filter_json_by_npy(self, json_data, data_dir):
        """
        Filter out rows in the JSON data where the corresponding npy file is missing.

        Parameters:
        - json_data: List of dictionaries, each containing a "ytid" field.
        - data_dir: Directory where the npy files are located.

        Returns:
        - Filtered list of dictionaries where corresponding npy files are found.
        """
        npy_folder = os.path.join(data_dir, 'npy')
        available_npy = {
            fname.partition('[')[-1].partition(']')[0]: fname
            for fname in os.listdir(npy_folder)
            if fname.endswith(".npy")
        }
        filtered_data = []
        missing_npys_cnt = 0

        for split in ["train", "test"]:
            for row in list(json_data[split]):
                if row['ytid'] in available_npy:
                    row["split"] = split
                    filtered_data.append(row)
                else: 
                    # print(f"Found missing .npy file for {row['ytid']}. Removing...")
                    missing_npys_cnt += 1
        print(f"Total missing npys: {missing_npys_cnt}")
        return filtered_data, available_npy

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ytid = self.ann[idx]['ytid']
        npy_fname = self.npy_fnames[ytid]
        npy_path = os.path.join(self.data_dir, 'npy', f'{npy_fname}')
        raw_audio = np.load(npy_path)
        
        resampler = T.Resample(16000, self.resample_rate)
        audio_input = resampler(torch.from_numpy(raw_audio).float())
        
        audio = self.processor(audio_input, 
                               sampling_rate=self.resample_rate, 
                               return_tensors="pt")['input_values'][0]
        txt = [self.ann[idx]['caption_writing']]
        # return audio features, text answer, and text question
        return {'ytid': ytid, 'input': audio, 'text': txt, 'instruction': [" "]}

    def collater(self, samples):
        #padding to max length in a batch
        ytids = [s['ytid'] for s in samples]
        audios = [s['input'] for s in samples]
        audio_sizes = [len(s['input']) for s in samples]
        audio_size = max(audio_sizes)
        txts = [" ".join(s['text']) for s in samples]
        instructions = [" ".join(s['instruction']) for s in samples]

        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        attn_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(True)
        )

        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            else: #diff < 0
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                attn_mask[i, diff:] = False

        attn_mask = attn_mask.int()

        return {'ytid': ytids, 
                'input': collated_audios, 
                'text': txts, 
                'instruction': instructions,
                'attention_mask': attn_mask,
                }