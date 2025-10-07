# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import argparse
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50526, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id
        self.use_attention_mask = use_attention_mask
        
        # Pre tokenize to create attention masks if required
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["text"]
        ]
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]
        
        if self.use_attention_mask:
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None
            
    def _create_attention_mask(self, encoded_text):
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]
    
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        
        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
            
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        ) 
        
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
    
