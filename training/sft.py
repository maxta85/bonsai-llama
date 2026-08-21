"""Supervised fine-tuning (SFT) for chat/instruction following.

Converts instruction-response pairs into Qwen3 chat format, tokenizes them,
and yields batches with loss masking (only compute loss on assistant tokens,
not on user/system tokens).

Supported input formats:
  1. ShareGPT JSONL: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
  2. Alpaca JSONL: {"instruction": "...", "output": "..."}  (optional "input")
  3. OpenAssistant/HF messages: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output format (Qwen3 chat, no thinking mode):
  <|im_start|>system\n{system}<|im_end|>
  <|im_start|>user\n{user}<|im_end|>
  <|im_start|>assistant\n{assistant}<|im_end|>
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader


# Qwen3 chat template tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOL = "\n"


def format_chat_turn(role: str, content: str) -> str:
    """Format a single chat turn in Qwen3 format."""
    return f"{IM_START}{role}{EOL}{content}{IM_END}{EOL}"


def format_conversation(messages: list[dict], system: str = "You are a helpful assistant.") -> str:
    """Format a full conversation in Qwen3 chat format (no thinking mode).

    messages: list of {"role": "user"/"assistant"/"system", "content": "..."}
    If no system message is present, one is prepended.
    Returns the formatted string.
    """
    has_system = any(m["role"] == "system" for m in messages)
    parts = []
    if not has_system:
        parts.append(format_chat_turn("system", system))
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(format_chat_turn(role, content))
    return "".join(parts)


def sharegpt_to_messages(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT format to standard messages format.

    ShareGPT uses {"from": "human"/"gpt"/"system", "value": "..."}
    Standard uses {"role": "user"/"assistant"/"system", "content": "..."}
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system",
                "tool": "user", "function": "user"}
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "")
        if not content:
            continue
        # Skip tool output turns (they're not something we train on)
        if turn.get("from") == "tool":
            continue
        # Skip turns that look like tool calls (JSON with name/arguments)
        if role == "assistant" and content.strip().startswith("{") and '"name"' in content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def alpaca_to_messages(example: dict) -> list[dict]:
    """Convert Alpaca format to messages.

    Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]


def detect_format(line: dict) -> str:
    """Detect which format a JSONL line is in."""
    if "conversations" in line:
        return "sharegpt"
    if "messages" in line:
        return "messages"
    if "instruction" in line:
        return "alpaca"
    return "unknown"


def parse_example(line: dict, system: str = "You are a helpful assistant.") -> list[dict] | None:
    """Parse a single JSONL example into messages format.

    Returns list of {"role": ..., "content": ...} or None if unparseable.
    """
    fmt = detect_format(line)
    if fmt == "sharegpt":
        msgs = sharegpt_to_messages(line["conversations"])
    elif fmt == "messages":
        msgs = line["messages"]
    elif fmt == "alpaca":
        msgs = alpaca_to_messages(line)
    else:
        return None

    # Validate: need at least one user and one assistant turn
    has_user = any(m["role"] == "user" for m in msgs)
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    if not has_user or not has_assistant:
        return None

    # Override system if present in the data
    if fmt == "sharegpt":
        for m in msgs:
            if m["role"] == "system":
                system = m["content"]
                msgs.remove(m)
                break
    elif fmt == "messages" and msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"]
        msgs = msgs[1:]

    # Re-add system at front
    msgs = [{"role": "system", "content": system}] + msgs
    return msgs


class SFTDataset(IterableDataset):
    """Tokenize instruction-response pairs and yield with loss masking.

    Loss is only computed on assistant tokens (labels for non-assistant
    tokens are set to -100, the ignore_index for cross-entropy).
    """

    def __init__(self, examples: list[dict], tokenizer, seq_len=2048,
                 system: str = "You are a helpful assistant."):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.system = system

    def __iter__(self):
        for example in self.examples:
            messages = parse_example(example, system=self.system)
            if messages is None:
                continue

            # Build the full conversation string
            full_text = format_conversation(messages, system=self.system)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

            # Build labels: mask everything except assistant turns
            labels = [-100] * len(full_ids)

            # Re-tokenize incrementally to find assistant turn boundaries
            # Strategy: build up the text turn by turn, track token positions
            prefix = format_chat_turn("system", self.system)
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            pos = len(prefix_ids)

            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    continue
                turn_text = format_chat_turn(msg["role"], msg["content"])
                turn_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)

                if msg["role"] == "assistant":
                    # Unmask assistant tokens (train on these)
                    for j in range(pos, min(pos + len(turn_ids), len(labels))):
                        labels[j] = full_ids[j]

                pos += len(turn_ids)

            # Truncate to seq_len
            if len(full_ids) > self.seq_len:
                full_ids = full_ids[:self.seq_len]
                labels = labels[:self.seq_len]

            if not any(l != -100 for l in labels):
                continue  # skip if no assistant tokens to learn

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            label_ids = torch.tensor(labels, dtype=torch.long)
            yield {"input_ids": input_ids, "labels": label_ids}


def pad_collate(batch):
    """Pad sequences to equal length within a batch."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    pad_id = 151643  # Qwen3 pad token id

    padded_ids = []
    padded_labels = []
    attention_masks = []
    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                                          torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def load_sft_dataset(jsonl_path: str, tokenizer, batch_size=4, seq_len=2048,
                     system: str = "You are a helpful assistant.",
                     max_examples: int | None = None) -> DataLoader:
    """Load a local JSONL file and return a DataLoader for SFT.

    Supports ShareGPT, Alpaca, and messages formats (auto-detected per line).
    """
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_examples and len(examples) >= max_examples:
                break

    print(f"  loaded {len(examples)} examples from {jsonl_path}")
    ds = SFTDataset(examples, tokenizer, seq_len=seq_len, system=system)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=pad_collate, pin_memory=True, drop_last=False)


def load_alpaca(tokenizer, batch_size=4, seq_len=2048,
                max_examples: int | None = None) -> DataLoader:
    """Load Stanford Alpaca dataset (52K instruction pairs) from HuggingFace."""
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    examples = []
    for ex in raw:
        examples.append(ex)
        if max_examples and len(examples) >= max_examples:
            break
    print(f"  loaded {len(examples)} Alpaca examples")
    ds = SFTDataset(examples, tokenizer, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=pad_collate, pin_memory=True, drop_last=False)


def load_dolly(tokenizer, batch_size=4, seq_len=2048,
               max_examples: int | None = None) -> DataLoader:
    """Load Dolly 15K dataset from HuggingFace."""
    from datasets import load_dataset
    raw = load_dataset("databricks/databricks-dolly-15k", split="train",
                       streaming=True)
    examples = []
    for ex in raw:
        # Dolly uses instruction/response/category format
        examples.append({
            "instruction": ex.get("instruction", ""),
            "input": ex.get("context", ""),
            "output": ex.get("response", ""),
        })
        if max_examples and len(examples) >= max_examples:
            break
    print(f"  loaded {len(examples)} Dolly examples")
    ds = SFTDataset(examples, tokenizer, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=pad_collate, pin_memory=True, drop_last=False)


def load_openhermes(tokenizer, batch_size=4, seq_len=2048,
                    max_examples: int | None = None) -> DataLoader:
    """Load OpenHermes 2.5 (1M instruction pairs) from HuggingFace."""
    from datasets import load_dataset
    raw = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    examples = []
    for ex in raw:
        examples.append(ex)
        if max_examples and len(examples) >= max_examples:
            break
    print(f"  loaded {len(examples)} OpenHermes examples")
    ds = SFTDataset(examples, tokenizer, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=pad_collate, pin_memory=True, drop_last=False)
