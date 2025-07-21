from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights generated in chapter 5.
    """

    GPT_CONFIG_774M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1280,         # Embedding dimension
        "n_heads": 20,          # Number of attention heads
        "n_layers": 36,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": True       # Query-key-value bias - Older configs used bias.
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    
    model_path = Path("..") / ("my-llm-from-scratch") /"gpt2-large-774M.pth"
    if not model_path.exists():
        print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_774M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_774M


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()
