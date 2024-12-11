"""
Code for loading the MusicInstruct Short/Long dataset (https://huggingface.co/datasets/m-a-p/Music-Instruct)
Modified and adapted from https://github.com/zihaod/MusiLingo/blob/main/musilingo/datasets/datasets/cmi_dataset.py
"""
from utils.base_dataset import BaseDataset
import os
import torch
import numpy as np
import json
import random
import torchaudio.transforms as T

class CMIDataset(BaseDataset):
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

        with open(ann_path, 'r') as f:
            self.all_ann = json.load(f)
        self.all_ann = self.all_ann['QA']
        self.all_ann, self.npy_fnames = self.filter_json_by_npy(self.all_ann, self.data_dir)
        
        # train/val/test split
        self.ann = []
        if self.split == "all":
            # Include all annotations
            self.ann = self.all_ann
        elif self.split in ["train", "test"]:
            # Standard train/test split
            is_eval = self.split == "test"
            for item in self.all_ann:
                if item['is_audioset_eval'] == is_eval:
                    self.ann.append(item)
        elif self.split == "val":
            # New 'val' split: random subset of items with is_eval=True
            eval_items = [item for item in self.all_ann if item['is_audioset_eval']]
            
            if val_subset_size is not None and val_subset_size < len(eval_items):
                self.ann = random.sample(eval_items, val_subset_size)
            else:
                self.ann = eval_items

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
            fname.partition('.npy')[0]: fname
            for fname in os.listdir(npy_folder)
            if fname.endswith(".npy")
        }
        filtered_data = []
        missing_npys_cnt = 0
        for row in json_data:
            if row['ytid'] in available_npy:
                filtered_data.append(row)
            else: 
                # print(f"Found missing .npy file for {row['ytid']}. Removing...")
                missing_npys_cnt += 1
        print(f"Total missing npys: {missing_npys_cnt}")
        return filtered_data, available_npy
    
    def get_max_length(self):
        # Initialize variables to store the longest question and answer lengths
        max_question_length = 0
        max_answer_length = 0

        # Iterate through each dictionary in self.ann
        for ann in self.all_ann:
            # Check the length of the 'question' and 'answer' fields
            question_length = len(ann['question'])
            answer_length = len(ann['answer'])
            # Update the max lengths if the current lengths are greater
            max_question_length = max(max_question_length, question_length)
            max_answer_length = max(max_answer_length, answer_length)

        return max_question_length, max_answer_length

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
        
        instruction = [self.ann[idx]['question']]
        txt = [self.ann[idx]['answer']]
        # return audio features, text answer, and text question
        return {'ytid': ytid, 'input': audio, 'text': txt, 'instruction': instruction}

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