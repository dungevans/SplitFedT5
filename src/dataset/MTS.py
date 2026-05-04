import os
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Giữ nguyên các import cũ của bạn
# from src.dataset.GSM8K import GSM8K
# from src.dataset.EMOTION import EMOTIONDataset, load_train_EMOTION, load_test_EMOTION

class MTSDataset(Dataset):
    """Dataset tùy chỉnh cho bài toán MTS-Dialog với T5"""
    def __init__(self, dataframe, tokenizer, max_source_len=768, max_target_len=160):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Format đầu vào và đầu ra theo logic FlanT5Summarizer của bạn
        source_text = str(row['dialogue'])
        target_text = f"<Header> {row['section_header']} <Summary> {row['section_text']}"

        # Tokenize source
        source_en = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target (labels)
        target_en = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = target_en['input_ids'].squeeze()
        # Thay thế pad_token_id bằng -100 để không tính loss trên các token padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_en['input_ids'].squeeze(),
            'attention_mask': source_en['attention_mask'].squeeze(),
            'labels': labels
        }