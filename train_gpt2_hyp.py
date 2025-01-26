import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

import random
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import json

from dataclasses import dataclass
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast # type: ignore #

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
# import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.geoopt.optim import RiemannianSGD
from modules.layers import *
from modules.muon import *
from modules.loader import *

import argparse

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Muon optimizer

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


@dataclass
class Hyperparameters:
    # data hyperparams
    data_path : str = 'data/fineweb10B'
    input_bin: str = ''  
    input_val_bin: str = ''  
    num_vocab : int = 50304
    # optimization hyperparams
    batch_size : int = 64 # batch size, in sequences, across all devices
    device_batch_size : int = 32 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 1_000 # number of iterations to run (for FW 2.7B was 4578)
    cooldown_frac : float = 0.4
    weight_decay : float = 0
    # evaluation and logging hyperparams
    generate_every : int = 1_000
    train_loss_every : int = 10 
    val_loss_every : int = 10 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    # model
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    head_mode : str = 'euc'
    attn_mode : str = 'euc'
    curvature : float = 1.0
    learnable: bool = True
    k_lr: float = 0.
    seed : int = 0

    def __post_init__(self):
        # Dynamically set .bin paths based on data_path
        if 'tinystories' in self.data_path:
            self.input_bin = f"{self.data_path}/train.bin"
            self.input_val_bin = f"{self.data_path}/val.bin"
        elif 'fineweb' in self.data_path:
            self.input_bin = f"{self.data_path}/fineweb_train_*.bin"
            self.input_val_bin = f"{self.data_path}/fineweb_val_*.bin"
        else:
            raise ValueError("Specify proper data path")



# -----------------------------------------------------------------------------
# The main GPT-2 model
@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    head_mode : str = 'euc'
    attn_mode : str = 'euc'
    curvature : float = 1.0
    learnable : bool = True
    block_size : int = 1024

class GPT(nn.Module):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        if config.head_mode == 'euc':
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            stdv = 1. / math.sqrt(config.n_embd)
            nn.init.uniform_(self.lm_head.weight.data, -stdv, stdv)

        elif config.head_mode == 'hyp':
            self.manifold = CustomLorentz(k=torch.tensor([config.curvature]))
            self.lm_head = LorentzMLR(
                manifold=self.manifold,
                num_features=config.n_embd,
                num_classes=config.vocab_size
            )
        else:
            raise ValueError("Invalid head_mode, choose 'euc'/'hyp'.")

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def encode_text(self, text):
        """Encodes a string into token IDs."""
        return self.tokenizer.encode(text, return_tensors="pt").to(device)

    def decode_tokens(self, tokens):
        """Decodes token IDs into a readable string."""
        return self.tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)

    def generate_text(self, context, max_length=200, temperature=1.0, top_k=50):
        self.eval()
        generated = context.clone()
        for _ in range(max_length):
            with torch.no_grad():
                logits, _ = self(generated, return_logits=True)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated


# -----------------------------------------------------------------------------
# parser

parser = argparse.ArgumentParser(description="Train GPT model with customizable parameters.")

parser.add_argument(
    "--curvature",
    type=float,
    default=1.0,
    help="Set the curvature for the Lorentz manifold (float, default: 1.0); if learnable, then is initial value."
)
parser.add_argument("--learnable", type=bool, default=True, help="Is the curvature learnable? (default: False)")
parser.add_argument("--head_mode", type=str, default='euc', help="Set the mode for LM Head (default: 'euc')")
parser.add_argument("--attn_mode", type=str, default='euc', help="Set the mode for attention layers (default: 'euc')")
parser.add_argument("--data_path", type=str, default='data/fineweb10B', help="Path to dataset directory (default: 'data/tinystories')")
parser.add_argument("--num_iterations", type=int, default=4578, help="Number of iterations (default=4578)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument("--k_lr", type=float, default=1., help="Set the learning rate for curvature parameter.")
parser.add_argument("--head_dim", type=int, default=128, help="Set the attention head dimension of the GPT model.")
parser.add_argument("--n_head", type=int, default=6, help="Set the number of attention heads in the GPT model.")

args_from_cli = parser.parse_args()

args = Hyperparameters(
    seed=args_from_cli.seed,
    data_path=args_from_cli.data_path,
    num_iterations=args_from_cli.num_iterations,
    head_mode=args_from_cli.head_mode,
    attn_mode=args_from_cli.attn_mode,
    curvature=args_from_cli.curvature,
    learnable=args_from_cli.learnable,
    n_embd=args_from_cli.n_head*args_from_cli.head_dim,
    n_head=args_from_cli.n_head
)


# -----------------------------------------------------------------------------
# int main
SEED = args.seed  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  

if "tinystories" in args.data_path:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(args.data_path, "tinystories_tokenizer.json"),
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="[PAD]"
    )
else:
    # GPT-2 tokenizer for FineWeb
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token  

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
# num_vocab = 50304
model = GPT(GPTConfig(vocab_size=args.vocab_size, 
                      n_layer=args.n_layer, 
                      n_head=args.n_head,
                      n_embd=args.n_embd,
                      head_mode=args.head_mode,
                      attn_mode=args.attn_mode,
                      curvature=args.curvature,
                      learnable=args.learnable,
                      block_size=args.sequence_length))
model = model.cuda()
# if hasattr(config, "coordinate_descent_tuning"):
#     config.coordinate_descent_tuning = True # suggested by @Chillee
# model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

head_k_params, attn_k_params = [], []
for name, param in raw_model.named_parameters():
    if "manifold.k" in name:
        head_k_params.append(param)
    elif "attn.k" in name:
        attn_k_params.append(param)

k_params = head_k_params + attn_k_params
for p in k_params:
    p.requires_grad = args.learnable  

if master_process:
    print(f"k params lengths: head = {len(head_k_params)}, attn = {len(attn_k_params)}")
        
lm_head_params = [p for name, p in raw_model.lm_head.named_parameters() if (p.requires_grad and ("manifold.k" not in name))]

params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
wte_params = [raw_model.transformer.wte.weight]

optimizer_lm_head = RiemannianSGD(
    [{'params': lm_head_params}], lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True, stabilize=1
)

optimizer_muon = Muon(matrix_params, lr=0.05, momentum=0.95)

optimizer_wte = torch.optim.Adam(wte_params, lr=0.6, betas=(0.8, 0.95), fused=True)

if k_params:
    optimizer_k = torch.optim.SGD([
        {"params": head_k_params, "lr": 10.},  
        {"params": attn_k_params, "lr": 1.}  
    ], momentum=0.9, nesterov=True)
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte, optimizer_k]
    if master_process:
        print(f"k is learned, {args.learnable}")
else:
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte]
    if master_process:
        print(f"k is not learned, {args.learnable}")


def get_lr(it):
    t = 1 - it / args.num_iterations # time remaining in training
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
    return w * 1.0 + (1 - w) * 0.1
    
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# schedulers[-1] = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizers[-1],
#     mode='max',           # Monitor for the minimum metric (e.g., validation loss).
#     factor=0.1,           # Reduce LR by a factor of 0.1.
#     patience=5,           # Wait for 5 epochs without improvement.
#     verbose=True          # Print LR reduction messages.
# )

# begin logging
if master_process:

    now = datetime.datetime.now()
    date_part = now.strftime('%d.%m')  
    seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).seconds #int(time.time() % 86400)
    run_id = f"{date_part}_{seconds_since_midnight}"

    suffix = f"{args.head_mode}_head_{args.attn_mode}_attn"
    # Construct the new folder name
    run_id = f"{run_id}_{suffix}_{args.seed}"
    
    # Create log directory and file
    logdir = f'runs/{run_id}/'
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
    logfile = os.path.join(logdir, 'log.txt')
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_s = 0.0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
total_t0 = time.time()
train_loss_accum = 0.0
train_loss_count = 0
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_s = 0.0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
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
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000*training_time_s/(timed_steps-1):.0f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000*training_time_s/(timed_steps-1):.0f}ms\n')
            writer.add_scalar('Loss/Validation', val_loss.item(), step)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and step and (step % args.generate_every == 0):
        # Use a fixed prompt or context for generation
        prompt = "Once upon a time"  # Customize as per your dataset
        context = raw_model.encode_text(prompt)
        
        # Generate text
        generated_tokens = raw_model.generate_text(context, max_length=200, temperature=1.0, top_k=50)
        generated_text = raw_model.decode_tokens(generated_tokens[0])
        
        # Log the generated text to TensorBoard
        writer.add_text(f"Generated_Text/Step_{step}", generated_text, step)
        
        # Optionally log to console for immediate feedback
        print(f"[Step {step}] Generated Text: {generated_text}")


    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'ckpts/%s_state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
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


    for name, param in raw_model.named_parameters():
        if "manifold.k" in name:
            print(f"Parameter: {name} = {param.item():.2f}, Requires Grad: {param.requires_grad}")
            print(f"Gradient: {param.grad}")
    for name, p in model.named_parameters():
        if p.grad is None:
            # print(f"WARNING: Parameter {name} has no gradient. Skipping.")
            continue
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    train_loss_accum += train_loss.item()
    train_loss_count += 1
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process and (step+1) % args.train_loss_every == 0:# within the main training loop, after logging validation loss or training loss
        
        if k_params:  # Only log if curvature is learnable
            for i, param in enumerate(head_k_params):
                curvature_value = param.item()  
                # curvature_grad = param.grad
                if i == 0:
                    print(f"Head curvature value: {curvature_value:.2f}")
                    writer.add_scalar(f"Curvature/Head", curvature_value, step)
                    # writer.add_scalar(f"Curvature Gradient", curvature_grad.item(), step)
            for i, param in enumerate(attn_k_params):
                curvature_value = param.item()  
                print(f"Attn curvature {i}: {curvature_value}")
                writer.add_scalar(f"Curvature/Attn/{i}", curvature_value, step)

        avg_train_loss = train_loss_accum / train_loss_count
        elapsed_time = time.time() - total_t0
        approx_time = training_time_s + (time.time() - t0)
        avg_time_per_step = approx_time/timed_steps
        estimated_total_time = avg_time_per_step * args.num_iterations
        print(f"step:{step+1}/{args.num_iterations} avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}/{estimated_total_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms\n")
        writer.add_scalar('Loss/Train', avg_train_loss, step)
        train_loss_accum = 0.0
        train_loss_count = 0

if master_process:
    total_training_time = time.time() - total_t0
    print(f"Total training time: {total_training_time:.2f}s")
    with open(logfile, "a") as f:
        f.write(f"Total training time: {total_training_time:.2f}s\n")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if master_process:
    writer.close()
dist.destroy_process_group()
