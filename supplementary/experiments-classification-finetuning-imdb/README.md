# Results of Local Experiments - Classification of Sentiment of 50k IMDb Movie Reviews 


## Overview

This folder contains experiments focused on the comparison of a decoder-style GPT-2 model and encoder-style LLMS like [BERT (2018)](https://arxiv.org/abs/1810.04805), [RoBERTa (2019)](https://arxiv.org/abs/1907.11692), and [ModernBERT (2024)](https://arxiv.org/abs/2412.13663).

The work is based on my localized versions of [Sebastian Raschka's experiments](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/03_bonus_imdb-classification), which utilze the [50k IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) with a binary classification objective predicting whether a reviewer liked or disliked a movie.


## Experiment Results

|       | Model                        | Test accuracy |
| ----- | ---------------------------- | ------------- |
| **1** | Logistic Regression Baseline | 88.77%        |
| **2** | 124M GPT-2 Baseline          | ------        |
| **3** | 340M BERT                    | 91.43%        |
| **4** | 66M DistilBERT               | 90.53%        |
| **5** | 355M RoBERTa                 | 92.15%        |
| **6** | 304M DeBERTa-v3              | ------        |
| **7** | 149M ModernBERT Base         | ------        |
| **8** | 395M ModernBERT Large        | ------        |


