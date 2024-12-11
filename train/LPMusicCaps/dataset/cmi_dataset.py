"""
Code for loading the MusicInstruct Short/Long dataset (https://huggingface.co/datasets/m-a-p/Music-Instruct)
Modified and adapted from https://github.com/zihaod/MusiLingo/blob/main/musilingo/datasets/datasets/cmi_dataset.py
"""
import os
import torch
import numpy as np
import json
import random
import torchaudio.transforms as T
from torch.utils.data import Dataset

class CMIDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            ann_path, 
            split, 
            val_subset_size=None,
            inflate=False,
            sr=16000, 
            duration=10,
        ):
        super().__init__()
        self.split = split # train or test
        self.data_dir = data_dir #music_data
        self.n_samples = int(sr * duration)

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
    
    def load_audio(self, npy_path):
        audio = np.load(npy_path, mmap_mode='r')

        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples)
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        random_idx = random.randint(0, audio.shape[-1]-self.n_samples)
        audio_tensor = torch.from_numpy(np.array(audio[random_idx:random_idx+self.n_samples]).astype('float32'))
        return audio_tensor

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ytid = self.ann[idx]['ytid']
        npy_fname = self.npy_fnames[ytid]
        npy_path = os.path.join(self.data_dir, 'npy', f'{npy_fname}')
        audio_tensor = self.load_audio(npy_path)
        
        instruction = [self.ann[idx]['question']]
        txt = [self.ann[idx]['answer']]
        # return audio features, text answer, and text question
        return ytid, txt, audio_tensor, instruction
    
    def collater(self, batch):
        """
        Collates a batch of data returned by the dataset's __getitem__ method.

        Parameters:
            batch: List of tuples (ytid, txt, audio_tensor, instruction)

        Returns:
            Collated batch as a tuple of batched ytids, texts, audio tensors, and instructions.
        """
        ytids = [item[0][0] for item in batch]
        txts = [item[1][0] for item in batch]  # list of lists (text answers)
        audio_tensors = torch.stack([item[2] for item in batch])  # Stack audio tensors
        instructions = [item[3][0] for item in batch]  # list of lists (questions)
        
        return ytids, txts, audio_tensors, instructions
