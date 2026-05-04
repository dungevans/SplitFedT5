import os
import json
import random

from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
from datasets import load_dataset

from src.dataset.GSM8K import GSM8K
from src.dataset.EMOTION import EMOTIONDataset
from src.dataset.EMOTION import load_train_EMOTION
from src.dataset.EMOTION import load_test_EMOTION
from src.dataset.MTS import MTSDataset
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
def dataloader(model_name =None, data_name=None, batch_size=None, distribution=500, train=True):
    if data_name == 'GSM8K':
        if model_name == 'GPT2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_name == 'Llama':
            tokenizer = AutoTokenizer.from_pretrained('JackFram/llama-160m')
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if train:
            path = os.path.join("data/", f"train.jsonl")

            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            random.shuffle(data)

            train_set = data[:distribution]
            for ex in train_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(train_set)} train examples")

            train_set = GSM8K(tokenizer, train_set, False)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            path = os.path.join("data/", f"test.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            random.shuffle(data)

            test_set = data[:500]
            for ex in test_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(test_set)} test examples")
            test_set = GSM8K(tokenizer, test_set,False)
            test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
            return test_loader

    if data_name == 'EMOTION':
        dataset = load_dataset(
            'ag_news',
            download_mode='reuse_dataset_if_exists',
            cache_dir='./hf_cache'
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        if train:
            num_label = int(distribution / 4)
            distribution = [num_label, num_label, num_label, num_label]
            train_texts, train_labels = load_train_EMOTION(dataset, distribution)
            train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            test_texts, test_label = load_test_EMOTION(2000, dataset)
            test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
            test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
            return test_loader
    if data_name == 'MTS-Dialog' or data_name == 'T5-CSV':
        model_path = "philschmid/flan-t5-base-samsum"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Đọc file train.csv từ thư mục gốc
        csv_path = os.path.join(os.path.dirname(__file__), "train.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file {csv_path} tại thư mục gốc.")
        
        df = pd.read_csv(csv_path)
        
        # Chia train/test nếu bạn chỉ có 1 file train.csv
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        
        if train:
            # Giới hạn số lượng mẫu dựa trên biến distribution nếu cần
            train_subset = train_df.head(distribution) if distribution < len(train_df) else train_df
            print(f"[T5] {len(train_subset)} train examples loaded.")
            
            dataset = MTSDataset(train_subset, tokenizer)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            print(f"[T5] {len(test_df)} test examples loaded.")
            dataset = MTSDataset(test_df, tokenizer)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)