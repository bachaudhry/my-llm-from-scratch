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


### --- LoRA layers ---

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
  
### --- Dataset handling ---
  
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
      
      
def download_and_unzip(url, zip_path, extract_to, new_file_path):
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Renaming the file to indicate its format
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
  

def create_dataset_csvs(new_file_path):
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])
    
    # Create balanced dataset 
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"]  == "ham"].sample(n_spam, random_state=42)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # Sample and save csv files
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
  

### --- Model pipeline

def instantiate_model(choose_model, load_weights):
    """
    Defaults to GPT-2 Medium unless specified.
    """
    BASE_CONFIG = {
          "vocab_size": 50257,     # Vocabulary size
          "context_length": 1024,  # Context length
          "drop_rate": 0.0,        # Dropout rate
          "qkv_bias": True         # Query-key-value bias
      }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(42)
        
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load("gpt2/gpt2-medium-355M.pth", weights_only=True))

    if load_weights:
        model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()
    return model
  
  
  
def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    if trainable_token_pos == "flexible": # Select the last tokens before the padding tokens
        # Refer to discussion https://github.com/rasbt/LLMs-from-scratch/discussions/434
        # Find the last non-padding token for each sequence in the batch
        pad_token_id = 50256 #<|endoftext|> token
        mask = input_batch != pad_token_id
        # Get position of the last real token
        last_token_pos = mask.sum(dim=1) - 1
        
        # Get model outputs
        logits = model(input_batch) # shape: [batch_size, seq_len, num_classes]
        
        # Select the logits corresponding to the last real token from each sequence
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]
        
        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss
    
    else:
        model_output = model(input_batch)
        if average_embeddings: 
            # Average over the sequence dimension i.e. dim=1
            logits = model_output.mean(dim=1)
        else:
            # Select embeddings at the specified token position
            logits = model_output[:, trainable_token_pos, :]
        
        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss
    
    
def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, 
                ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad() # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                
                # Find the last non-padding token for each sequence in the batch
                pad_token_id = 50256 #<|endoftext|> token
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1 # Position of the last real token
                
                logits = model(input_batch) # Logits of the last output token
                # Selec the logits corresponding to the last real token of each sequence
                batch_size = logits.size(0)
                selected_logits = logits[torch.arange(batch_size), last_token_pos]
                predicted_labels = torch.argmax(selected_logits, dim=-1)
                
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
            
    else:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                 
                model_output = model(input_batch)
                if average_embeddings:
                    #Average over the sequence dimension (dim=1)
                    logits = model_output.mean(dim=1)   
                else:
                    # Select embeddings at the specified token position
                    logits = model_output[:, trainable_token_pos, :]
                    
                predicted_labels = torch.argmax(logits, dim=-1)
                
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()
    return train_loss, val_loss

                
                    

            
        