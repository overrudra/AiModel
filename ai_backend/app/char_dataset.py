import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.inputs = [tokens[i:i+block_size] for i in range(len(tokens) - block_size)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx])
        target_ids = input_ids.clone()
        return input_ids, target_ids
