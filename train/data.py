import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional


class LeetCodeDistillDataset(Dataset):
    """LeetCode problems formatted for knowledge distillation.

    Each sample: system_prompt + problem → solution
    Tokenized with teacher tokenizer, padded to max_length.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 2048,
        dataset_name: str = "justindal/leetcode-python-dataset",
        streaming: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        ds = load_dataset(
            dataset_name, split=split, streaming=streaming
        )
        for item in ds:
            messages = item["messages"]
            if len(messages) >= 3:
                system_msg = messages[0]["content"]
                user_msg = messages[1]["content"]
                assistant_msg = messages[2]["content"]

                prompt = system_msg + "\n\n" + user_msg
                full_text = prompt + assistant_msg
                self.samples.append(
                    {
                        "prompt": prompt,
                        "completion": assistant_msg,
                        "full_text": full_text,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize full text (prompt + completion)
        full_encoding = self.tokenizer(
            sample["full_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize just the prompt to find where completion starts
        prompt_encoding = self.tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["attention_mask"].sum().item()

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)

        # Labels: -100 for prompt tokens, actual ids for completion
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloaders(
    tokenizer,
    batch_size: int = 1,
    max_length: int = 2048,
    num_workers: int = 0,
    dataset_name: str = "justindal/leetcode-python-dataset",
):
    train_dataset = LeetCodeDistillDataset(
        tokenizer, split="train", max_length=max_length, dataset_name=dataset_name
    )
    val_dataset = LeetCodeDistillDataset(
        tokenizer, split="valid", max_length=max_length, dataset_name=dataset_name
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
