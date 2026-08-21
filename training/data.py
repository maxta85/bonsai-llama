"""Dataset pipeline for distillation training.

Loads a HF dataset, tokenizes it, packs into fixed-length sequences, and
yields batches suitable for next-token-prediction distillation.

Default dataset is WikiText-103-raw-v1 (free, ~500MB, downloads in ~30s).
Swap for any HF dataset by name.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset, DataLoader


class PackedTextDataset(IterableDataset):
    """Tokenize a text dataset and pack into fixed-length sequences.

    Streams the dataset, tokenizes on the fly, and concatenates tokens into
    fixed-length chunks of `seq_len`. Each sample is one chunk; the label is
    the same chunk shifted by one (standard next-token prediction).
    """

    def __init__(self, dataset, tokenizer, seq_len=1024, split="train",
                 text_key="text", max_tokens=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.text_key = text_key
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        n_emitted = 0
        for example in self.dataset:
            text = example[self.text_key]
            if not text or not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
                n_emitted += 1
                if self.max_tokens and n_emitted >= self.max_tokens:
                    return


def make_dataloader(dataset, tokenizer, batch_size=4, seq_len=1024,
                    split="train", text_key="text", max_tokens=None,
                    num_workers=0):
    """Build a DataLoader over a packed-text dataset."""
    ds = PackedTextDataset(dataset, tokenizer, seq_len=seq_len, split=split,
                           text_key=text_key, max_tokens=max_tokens)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, drop_last=True)


def load_wikitext(tokenizer, batch_size=4, seq_len=1024, split="train",
                  max_tokens=None):
    """Load WikiText-103-raw-v1 and return a DataLoader."""
    from datasets import load_dataset
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                       split=split, streaming=True)
    return make_dataloader(raw, tokenizer, batch_size=batch_size,
                           seq_len=seq_len, split=split, max_tokens=max_tokens)


def load_dataset_by_name(name, tokenizer, batch_size=4, seq_len=1024,
                         split="train", text_key="text", max_tokens=None):
    """Load any HF dataset by name. Returns a DataLoader."""
    from datasets import load_dataset
    raw = load_dataset(name, split=split, streaming=True)
    return make_dataloader(raw, tokenizer, batch_size=batch_size,
                           seq_len=seq_len, split=split, text_key=text_key,
                           max_tokens=max_tokens)
