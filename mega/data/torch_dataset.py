import torch
from torch.utils.data import Dataset
import numpy as np
import random

np.random.seed(0)

random.seed(0)

torch.manual_seed(0)

torch.set_default_dtype(torch.float32)


class PromptDataset(Dataset):
    def __init__(self, prompts, model, tokenizer, device="cuda", **tokenizer_args):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_len = self.get_max_len()
        self.device = device
        self.model = model
        self.tokenizer_args = tokenizer_args
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.prompts)

    def get_max_len(self):
        return max(len(self.tokenizer(text)["input_ids"]) for text in self.prompts)

    def __getitem__(self, i):
        text = self.prompts[i]
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=min(self.model.config.max_position_embeddings, self.max_len),
            add_special_tokens=True,
        )

        encoding = {k: v[0].to(self.device) for k, v in encoding.items()}
        # print("encoded")
        return encoding
