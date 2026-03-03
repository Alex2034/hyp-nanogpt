"""
Загрузка чекпоинта 65280_fwe_eh_lr1_38.3M_s0_20000.pt и инференс.
Модель обучена на finewebedu, head_mode=euc, attn_mode=hyp.
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from transformers import GPT2TokenizerFast

from model.model import GPT
from model.config import Config

CHECKPOINT_PATH = Path("~/hypgpt/checkpoints/65280_fwe_eh_lr1_38.3M_s0_20000.pt")

config = Config(data_path="data/finewebedu")
config.n_heads = 16
config.n_layers = 16
config.head_dim = 16
config.n_embd = 256 
config.sequence_length = 1024
config.head_mode = "euc"
config.attn_mode = "hyp"
config.curvature = 1.0
config.k_lr = 1.0
config.vocab_size = 50257  # GPT-2

# GPT-2 tokenizer
_tokenizer_path = _repo_root / "data" / "gpt2_tokenizer"
if _tokenizer_path.exists():
    tokenizer = GPT2TokenizerFast.from_pretrained(str(_tokenizer_path))
else:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

model = GPT(config)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
state_dict = ckpt["model"]
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

prompt = "The meaning of life is"
context = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
generated = model.generate_text(context, max_length=150, temperature=0.8, top_k=50)
decoded = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

print(f"Prompt: {prompt!r}")
print(f"Generated:\n{decoded}")
