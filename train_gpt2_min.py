import os
import sys
import random
import datetime
import time
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast # type: ignore #

from model.rsgd import RiemannianSGD
from model.model import GPT
from utils.muon import Muon
from utils.loader import DistributedDataLoader
from utils.config import Config
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()

# Configurable arguments
parser.add_argument("--test_run", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--data_path", type=str, default="data/shakespeare_char")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device_batch_size", type=int, default=32)
parser.add_argument("--num_iterations", type=int, default=4)
parser.add_argument("--gen_every", type=int, default=0)
parser.add_argument("--train_loss_every", type=int, default=2)
parser.add_argument("--val_loss_every", type=int, default=2)
parser.add_argument("--head_dim", type=int, default=16)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sequence_length", type=int, default=128)
parser.add_argument("--k_lr", type=float, default=0.0)
parser.add_argument("--curvature", type=float, default=1.0)
parser.add_argument("--head_mode", type=str, default="euc", \
    help="Set the mode for LM Head")
parser.add_argument("--attn_mode", type=str, default="euc", \
    help="Set the mode for attention layers")

args = parser.parse_args()
config = Config(**vars(args))

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

if "shakespeare" in config.data_path:
    from data.shakespeare_char.CharTokenizer import CharacterTokenizer
    dataset_name = "shakespeare_char"
    tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")
    config.vocab_size = tokenizer.vocab_size
elif "tinystories_char" in config.data_path:
    dataset_name = "tinystories_char"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tinystories_char/char_tokenizer.json")
    config.vocab_size = tokenizer.vocab_size
elif "tinystories" in config.data_path:
    dataset_name = "tinystories"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(config.data_path, "tinystories_tokenizer.json"),
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    config.vocab_size = tokenizer.vocab_size
elif "fineweb" in config.data_path:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    if "finewebedu" in config.data_path:
        dataset_name = "finewebedu"
    else:
        dataset_name = "fineweb"
else:
    raise ValueError("Incorrect data_path")

def encode_text(tokenizer, text, device):
    return tokenizer.encode(text, return_tensors="pt").to(device)

def decode_tokens(tokenizer, tokens):
    # For character-level tokenizer, join characters without spaces
    if "char" in config.data_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    # For word-level tokenizers, use normal decoding
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)

# DDP setup
assert torch.cuda.is_available(), "CUDA is required for DDP but not available."
try:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
except KeyError as e:
    raise RuntimeError(f"Missing environment variable for DDP: {e}")
dist.init_process_group(backend='nccl')
    
device = torch.device(f'cuda:{ddp_local_rank}')
torch.cuda.set_device(device)

print(f"[Rank {ddp_rank}] Using device: {device}")

master_process = (ddp_rank == 0)
B, T = config.device_batch_size, config.sequence_length

assert config.batch_size % (B * ddp_world_size) == 0, "batch_size must be divisible by global batch size."
train_accumulation_steps = config.batch_size // (B * ddp_world_size)
tokens_per_iter = config.batch_size * config.sequence_length

train_loader = DistributedDataLoader(config.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(config.input_val_bin, B, T, ddp_rank, ddp_world_size)
val_steps = int(config.val_tokens_frac * val_loader.ntok_total) // (B * T * ddp_world_size)


if master_process:
    print(f"Training DataLoader: {train_loader.ntok_total / 1e6:.2f}M tokens across {len(train_loader.files)} files.")
    print(f"Validation DataLoader: {val_loader.ntok_total / 1e6:.2f}M tokens across {len(val_loader.files)} files.")
x, y = train_loader.next_batch()

    
# def register_hooks(model):
#     for name, module in model.named_modules():
#         if hasattr(module, 'weight'):
#             module.register_forward_hook(lambda m, i, o: print(f"[FWD] {name} output norm: {o.norm().item()}"))
#             module.register_backward_hook(lambda m, grad_input, grad_output: print(f"[BWD] {name} grad_output norm: {grad_output[0].norm().item()}"))

# Model setup
model = GPT(config)  
model = model.to(device)    

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

model = torch.compile(model)

end_event.record()
torch.cuda.synchronize()

compile_time = start_event.elapsed_time(end_event)
print(f"Model compiled in {compile_time:.1f}ms")

model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module  # Always access raw model via .module

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)

if master_process:
    print(f"Tokenizer vocab size: {config.vocab_size}")

params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
non_matrix_params = [p for p in params if p.ndim != 2]  
wte_params = [raw_model.transformer.wte.weight]

optimizer_wte = torch.optim.Adam(wte_params + non_matrix_params, lr=config.wte_lr, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer_muon = Muon(matrix_params, lr=config.muon_lr, momentum=0.95)

if config.head_mode == 'hyp':
        optimizer_head = RiemannianSGD(raw_model.lm_head.optim_params(), lr=config.head_lr)
else:  # Euclidean head
    optimizer_head = optim.SGD(raw_model.lm_head.parameters(), lr=config.head_lr)

optimizers = [optimizer_head, optimizer_muon, optimizer_wte]

init_lr = 1.0
end_lr  = 0.1
def get_lr(it):
    t = max(0, min(1, 1 - it/config.num_iterations))
    w = min(t / config.cooldown_frac, 1.0)
    return w * init_lr + (1 - w) * end_lr
    
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

if master_process:
    model_size = raw_model.model_size()
    print("\n=== Model ===")
    print(f"Model Size:    {model_size}\n")
    print(f"Data Path:            {config.data_path}")
    print(f"Sequence Length:      {config.sequence_length}")
    print(f"Batch Size (global):  {config.batch_size}")
    print(f"Batch Size (device):  {config.device_batch_size}")
    print(f"n_layers:              {config.n_layers}")
    print(f"n_heads:               {config.n_heads}")
    print(f"head_dim:             {config.head_dim}")
    print(f"n_embd:               {config.n_embd}")
    print("\n=== Experiment ===")
    print(f"Head mode:             {config.head_mode}")
    print(f"Attention mode:        {config.attn_mode}")
    print(f"Init curvature:        {config.curvature}")
    print(f"Curvature learning rate: {config.k_lr}")
    print(f"Seed:                 {config.seed}")
    print("==============================\n")


# begin logging
if master_process:
    def create_run_id(config, dataset_name, timestamp):
        """Create a run identifier."""
        # Aliases for common configurations
        dataset_aliases = {
            'shakespeare_char': 'sh',
            'tinystories_char': 'tsc',
            'tinystories': 'ts',
            'fineweb': 'fw'
        }
        mode_aliases = {
            'euc': 'e',
            'hyp': 'h'
        }
        
        # Get date and time components
        date = timestamp.strftime('%m.%d') 
        seconds_since_midnight = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).seconds
        
        # Get architecture configuration
        head, attn = mode_aliases[config.head_mode], mode_aliases[config.attn_mode]
        arch = f"{head}{attn}"
        
        # Build the hyperbolic parameters string if needed
        hyp_params = ""
        if 'h' in arch:
            hyp_params = f"_k{config.curvature}"
            if config.k_lr:
                hyp_params += f"_lr{config.k_lr:.0e}"  
        
        # Combine all components
        run_id = f"{seconds_since_midnight}_{dataset_aliases[dataset_name]}_{arch}{hyp_params}_s{config.seed}"
        return date, run_id

    # Create the run ID
    now = datetime.datetime.now()
    date, run_id = create_run_id(config, dataset_name, now)
    # Create log directory and file
    logdir = f'runs/{date}/{run_id}/'
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "tensorboard_logs"), exist_ok=True)

    print(f"Logs for this run will be stored in: {logdir}")

    print("Writing logs to: " + os.path.join(logdir, "tensorboard_logs"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard_logs"))

    config_path = os.path.join(logdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    def pretty_json(hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    writer.add_text("run_params", pretty_json(vars(args)))


# Initialize CUDA events
total_start_event = torch.cuda.Event(enable_timing=True)
interval_start_event = torch.cuda.Event(enable_timing=True)
interval_end_event = torch.cuda.Event(enable_timing=True)
intervals = []
# Record the start of total training time
total_start_event.record()
interval_start_event.record()  

train_loss_accum = 0.0
train_log_count = 0

val_loss_accum = 0.0
val_log_count  = 0

# begin training
train_loader.reset()
for step in range(config.num_iterations + 1):
    last_step = (step == config.num_iterations)
    
    if (last_step or (config.val_loss_every > 0 and step % config.val_loss_every == 0)):
        
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_loss_accum += val_loss
        val_log_count  += 1
        # log val loss to console 

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step

    for name, p in model.named_parameters():
        if p.grad is None:
            # print(f"WARNING: Parameter {name} has no gradient. Skipping.")
            continue
        p.grad /= train_accumulation_steps

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
        
    model.zero_grad(set_to_none=True)
    train_loss_accum += train_loss.item()
    train_log_count += 1
    
    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process and step % config.train_loss_every == 0:# within the main training loop, after logging validation loss or training loss
        
        interval_end_event.record()
        torch.cuda.synchronize()

        # Calculate elapsed time in milliseconds
        interval_time_ms = interval_start_event.elapsed_time(interval_end_event)
        intervals.append(interval_time_ms)
        
        avg_time_per_step = sum(intervals[-10:])
        estimated_total_time = avg_time_per_step * (config.num_iterations - step) / config.train_loss_every / 1e4
        
        # compute the averages
        avg_train_loss = train_loss_accum / max(1, train_log_count)
        avg_val_loss   = val_loss_accum   / max(1, val_log_count)

        # log
        tokens_seen = step * tokens_per_iter
        writer.add_scalar('Loss/Train',      avg_train_loss, tokens_seen)
        writer.add_scalar('Loss/Validation', avg_val_loss,   tokens_seen)
        print(f"step {step} ({interval_time_ms:.0f}ms): {tokens_seen/1e6:.2f}M tokens seen, train loss = {avg_train_loss:.4f}, val loss = {val_loss:.4f}, ETA = {estimated_total_time:.0f}s")
        
        # reset accumulators
        train_loss_accum = 0.0
        train_log_count  = 0
        val_loss_accum   = 0.0
        val_log_count    = 0

    
        if master_process and (last_step or (config.save_every > 0 and step % config.save_every == 0)):
            # save the state of the training process
            log = dict(step=step, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, 'ckpts/%s_state_step%06d.pt' % (run_id, step))
            # start the clock again

        if config.gen_every and master_process and ((step) % config.gen_every == 0):
            # Use a fixed prompt or context for generation
            prompt = "Once upon a time in a"  # Customize as per your dataset
            context = encode_text(tokenizer,prompt, device)
            
            # Generate text
            generated_tokens = raw_model.generate_text(context, max_length=config.gen_lenght, temperature=1.0, top_k=50)
            generated_text = decode_tokens(tokenizer, generated_tokens[0])
            
            # Log the generated text to TensorBoard
            writer.add_text(f"Generated_Text/Step_{step}", generated_text, step)
            
            # Optionally log to console for immediate feedback
            print(f"\nGenerated Text: \n{generated_text}\n")
        
        interval_start_event.record()
        
if master_process:
    total_end_event = torch.cuda.Event(enable_timing=True)
    total_end_event.record()
    torch.cuda.synchronize()

    total_time_s = total_start_event.elapsed_time(total_end_event) / 1e3
    print(f"Total training time: {total_time_s:.2f}s")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if master_process:
    writer.close()

dist.destroy_process_group()
