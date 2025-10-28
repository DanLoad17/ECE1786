import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed
import datasets

set_seed(1234)


class SentiDataset(Dataset):
    def __init__(self, ds_choice="small", split="train", truncation=-1):
        self.ds_choice = ds_choice
        self.truncation = truncation
        sst = datasets.load_dataset('glue', 'sst2')
        first1200 = sst['train'][:1200]

        train_sent, val_sent, train_label, val_label = train_test_split(
            first1200['sentence'],
            first1200['label'],
            test_size=0.2,
            shuffle=True)

        if split == "train":
            raw_data, label = train_sent, train_label
        else:
            raw_data, label = val_sent, val_label

        self.tokenizer = BPETokenizer()
        self.data = []   # List of tokenized sentences
        self.label = []  

        for x, y in zip(raw_data, label):
            tokenized = self.tokenizer(x).view(-1)  # Convert to a 1D tensor

            # Apply truncation if enabled
            if truncation >= 0:
                tokenized = tokenized[:truncation]
            self.data.append(tokenized)
            self.label.append(torch.tensor(y).float())

        # Set maximum sequence length (512 to match GPT block size)
        self.max_sentence_length = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x, y)

    def get_vocab_size(self):
        return 50257

    def get_block_size(self):
        return self.max_sentence_length
