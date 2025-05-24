#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def save_with_header(path: Path, ids: np.ndarray):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(ids)  # token count
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(ids.tobytes())

def tokenize_split(tokenizer, texts, split_name):
    print(f"[+] Tokenizing {split_name} ({len(texts)} examples)")
    all_ids = []
    for txt in tqdm(texts, desc=split_name, unit="ex"):
        ids = tokenizer.encode(txt, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        all_ids.append(np.array(ids, dtype=np.uint16))
    return np.concatenate(all_ids)

def main():
    out_dir = Path(__file__).parent
    print("[+] Loading TinyStories dataset")
    ds_train = load_dataset("roneneldan/TinyStories", split="train")
    ds_val   = load_dataset("roneneldan/TinyStories", split="validation")

    print("[+] Loading GPT-2 tokenizer")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    train_ids = tokenize_split(tok, ds_train["text"], "train")
    val_ids   = tokenize_split(tok, ds_val["text"],   "val")

    print("[+] Computing token frequencies on train split")
    freq = Counter(train_ids.tolist())
    freq_path = out_dir / "freq.json"
    with open(freq_path, "w") as f:
        json.dump(freq, f, indent=2)
    print(f"    • {len(freq)} unique tokens → saved to {freq_path.name}")

    print("[+] Writing binary files")
    save_with_header(out_dir / "train.bin", train_ids)
    save_with_header(out_dir / "val.bin",   val_ids)
    print("✅ Done.")

if __name__ == "__main__":
    main()
