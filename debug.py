import torch
from data.shakespeare_char.CharTokenizer import CharacterTokenizer
from model.model import GPT
from model.lorentz import LorentzDot
from utils.config import Config

# Initialize tokenizer
tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")

# Setup config
config = Config()
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

# Create dummy input using tokenizer
prompt = "Once upon a time in a"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Forward pass
logits, loss = model(input_ids)
print("Forward pass successful!")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss}")

loss.backward()
# # Try generation
# context = input_ids
# generated = model.generate_text(context, max_length=10)
# print("Generation successful!")
# print(f"Generated shape: {generated.shape}")
# print(f"Generated text: {tokenizer.decode(generated[0].tolist())}")

