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
      introduced by LoRA.
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