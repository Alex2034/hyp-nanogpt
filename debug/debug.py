import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT
from model.config import Config

# Initialize tokenizer
tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")

# Setup config 
config = Config(data_path="data/shakespeare_char")
config.n_layers = 2      
config.n_heads = 2       
config.sequence_length = 32
config.n_embd = 64      
config.batch_size = 1
config.head_mode = 'hyp'
config.attn_mode = 'euc'
config.vocab_size = tokenizer.vocab_size

# Create model
model = GPT(config)

# Create dummy input
prompt = "Once upon a time in a"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
idx = input_ids[:, :-1]
targets = input_ids[:, 1:].clone()

# Forward pass
logits, loss = model(idx, targets=targets)
print("Forward pass successful!")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss}")

loss.backward()

# Generation test
model.eval()
gen_prompt = "The "
gen_context = tokenizer.encode(gen_prompt, add_special_tokens=False, return_tensors="pt")
generated = model.generate_text(gen_context, max_length=20, temperature=0.8, top_k=40)
decoded = tokenizer.decode(generated[0].tolist())
print("Generation successful!")
print(f"Prompt: {gen_prompt!r}")
print(f"Generated: {decoded!r}")

