# This script serves to carry out experiments related to fine-tuning LLMs for classification tasks.
# It also introduces LoRA(Low Rank Adaptation) for parameter efficient fine-tuning, which can be extended
# to both classification and instruction fine-tuning tasks.


import argparse
import math
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from components import GPTModel
from gpt_generate import load_weights_into_gpt


class LoRALayer(torch.nn.Module):
    """
    
    - Rank: Governs the inner dimension of matrices A and B and determines the number of extra parameters
      introduced by LoRA. This is meant to create a balance between model adaptability and efficiency.
    - Alpha: Functions as a scaling factor for the output from LoRA. It determines the degree to which the
      output from the adapted layer can affect the original layer's output.
    - LoRA's typical goal is to substitue existing Linear layers, allowing weight updates to be applied 
      directly to the pretrained weights.
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
      
      
class LinearWithLoRA(torch.nn.Module):
    """
    Substitutes existing Linear layers with LoRA layers, thereby allowing weight updates to be applied 
    directly to the pre-existing pretrained weights. The replaced layers can include attention or 
    feed forward modules.
    """
    def __init__(self, linear, rank, alpha):
      super().__init__()
      self.linear = linear
      self.lora = LoRALayer(
        linear.in_features, linear.out_features, rank, alpha
      )
      
    def forward(self, x):
      return self.linear(x) + self.lora(x)
  
  
class LinearWithLoRAMerged(torch.nn.Module):
    """
    Equivalent code to LinearWithLoRA
    """
    def __init__(self, linear, rank, alpha):
      super().__init__()
      self.linear = linear
      self.lora = LoRALayer(
        linear.in_features, linear.out_features, rank, alpha
      )
      
    def forward(self, x):
      lora = self.lora.A @ self.lora.B
      combined_weight = self.linear.weight + self.lora.alpha * lora.T
      return torch.nn.functional.linear(x, combined_weight, self.linear.bias)
  
  
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
          self.data = pd.read_csv(csv_file)
          self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

          # Pre-tokenize texts
          self.encoded_texts = [
              tokenizer.encode(text)[:self.max_length]
              for text in self.data["Text"]
          ]

          if not no_padding:
              # Pad sequences to the longest sequence
              self.encoded_texts = [
                  et + [pad_token_id] * (self.max_length - len(et))
                  for et in self.encoded_texts
              ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        return max(len(encoded_text) for encoded_text in self.encoded_texts) # Pythonic!! ;)
      
      
