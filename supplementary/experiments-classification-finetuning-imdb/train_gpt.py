# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
from pathlib import Path
import time

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.gpt_download import download_and_load_gpt2
from utils.components import GPTModel, load_weights_into_gpt