import os
# import sys
import math
import random
import datetime
# import time
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2TokenizerFast  # type: ignore

from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.rsgd import RiemannianSGD
from model.model import GPT
from utils.muon import Muon
from utils.loader import DistributedDataLoader
from utils.config import Config
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()

# Configurable arguments
# parser.add_argument("--debug", action="store_true")
parser.add_argument("--data_path", type=str, default="data/shakespeare_char")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device_batch_size", type=int, default=32)
parser.add_argument("--num_iterations", type=int, default=4)
parser.add_argument("--gen_every", type=int, default=0)
parser.add_argument("--gen_prompt", type=str, default="Once ")
parser.add_argument("--gen_first", type=int, default=0)
parser.add_argument("--gen_length", type=int, default=200)
parser.add_argument("--train_loss_every", type=int, default=2)
parser.add_argument("--val_loss_every", type=int, default=2)
parser.add_argument("--log_curv_every", type=int, default=0)  # if 0,
# set equal to val_loss_every in post_init)
parser.add_argument("--save_every", type=int, default=0)
parser.add_argument("--head_dim", type=int, default=16)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sequence_length", type=int, default=128)
parser.add_argument("--k_lr", type=float, default=0.0)
parser.add_argument("--curvature", type=float, default=1.0)

parser.add_argument("--head_mode", type=str, default="euc")

parser.add_argument("--attn_mode", type=str, default="euc",
                    help="Set the mode for attention layers")

parser.add_argument("--print_multiplier", type=int, default=5)

args = parser.parse_args()
config = Config(**vars(args))

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

char_datasets = {
    "shakespeare_char", "tinystories_char", "taoteching", "cn_wiki"
    }

gpt2_datasets = {"tinystories", "fineweb", "finewebedu"}

dataset_name = os.path.basename(config.data_path)

if dataset_name in char_datasets:
    tokenizer = CharacterTokenizer.from_pretrained(
        save_directory=config.data_path
        )
    config.vocab_size = tokenizer.vocab_size

elif dataset_name in gpt2_datasets:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size

else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def encode_text(tokenizer, text, device):
    return tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
        ).to(device)


def decode_tokens(tokenizer, tokens):
    # For character-level tokenizer, join characters without spaces
    if "char" in config.data_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    # For word-level tokenizers, use normal decoding
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)


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

assert config.batch_size % (B * ddp_world_size) == 0, "batch_size must be \
    divisible by global batch size."
train_accumulation_steps = config.batch_size // (B * ddp_world_size)
tokens_per_iter = config.batch_size * config.sequence_length

train_loader = DistributedDataLoader(
    config.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(
    config.input_val_bin, B, T, ddp_rank, ddp_world_size)
val_steps = int(config.val_tokens_frac * val_loader.ntok_total) \
    // (B * T * ddp_world_size)

if master_process:
    print(
        f"Training DataLoader: {train_loader.ntok_total / 1e6:.2f}M tokens "
        f"across {len(train_loader.files)} files."
        )
    print(
        f"Validation DataLoader: {val_loader.ntok_total / 1e6:.2f}M tokens "
        f"across {len(val_loader.files)} files."
        )
    print(f"Tokenizer vocab size: {config.vocab_size}")

x, y = train_loader.next_batch()

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
raw_model = model.module

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)

# ---- 1. gather groups -------------------------------------------------------
curv_params = [blk.attn.log_c                   # or .log_c if you renamed it
               for blk in raw_model.transformer.h
               if hasattr(blk.attn, "log_c")]   # keep only blocks in hyp-mode

curv_id = {id(p) for p in curv_params}        # for fast membership testing

matrix_params, non_matrix_params = [], []
for p in raw_model.transformer.h.parameters():
    if id(p) in curv_id:                      # already in curvature group
        continue
    (matrix_params if p.ndim == 2 else non_matrix_params).append(p)


wte_params = [raw_model.transformer.wte.weight]


if config.head_mode == 'hyp':
    param_groups = raw_model.lm_head.optim_params()
    optimizer_head = RiemannianSGD(param_groups, lr=config.head_lr)

elif config.head_mode == 'euc':
    param_groups = [{'params': raw_model.lm_head.parameters()}]
    optimizer_head = torch.optim.Adam(
        [p for p in raw_model.lm_head.parameters()],
        lr=config.head_lr,
        betas=(0.8, 0.95),
        eps=1e-10,
        fused=True,
    )
else:
    raise ValueError("Invalid head_mode, choose 'hyp' or 'euc'")

# now flatten all parameter iterators into one list of Tensors
head_params = []
for grp in param_groups:
    head_params.extend(list(grp['params']))


optimizer_wte = torch.optim.Adam(wte_params + non_matrix_params,
                                 lr=config.wte_lr, betas=(0.8, 0.95),
                                 eps=1e-10, fused=True)
optimizer_muon = Muon(matrix_params, lr=config.muon_lr, momentum=0.95)

optimizers = [optimizer_head, optimizer_muon, optimizer_wte]

init_lr = 1.0
end_lr = 0.1


def get_lr(it):
    t = max(0, min(1, 1 - it/config.num_iterations))
    w = min(t / config.cooldown_frac, 1.0)
    return w * init_lr + (1 - w) * end_lr


schedulers = [
    torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers
    ]

if len(curv_params):
    optimizer_curv = torch.optim.SGD(curv_params, lr=config.k_lr, momentum=0.0)
    optimizers.append(optimizer_curv)
    schedulers.append(
        torch.optim.lr_scheduler.LambdaLR(optimizer_curv, lambda _: 1.0)
        )


def print_curvature_stats(blocks, grads):
    curvatures = []
    for block in blocks:
        if hasattr(block.attn, "log_c"):  # Only for blocks with curvature
            c = torch.exp(
                block.attn.log_c.detach().cpu()  # shape: (1, n_heads, 1, 1)
                )
            curvatures.append(c.reshape(-1))
    if not curvatures:
        print("No learnable curvatures found.")
        return
    all_c = torch.cat(curvatures)
    mean = all_c.mean().item()
    std = all_c.std(unbiased=False).item()
    stats = (
        f"Curvature stats over all blocks/heads: "
        f"{mean:.4g} ± {std:.2g} "
        f"(min={all_c.min().item():.3g}, max={all_c.max().item():.3g})"
    )
    print(f"\n{stats}\n{grads}\n")


def n_params(group):
    return sum(p.numel() for p in group)


def grad_norm(params, norm_type=2):
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in params]
            ), norm_type)
    return norm.item()


def log_curvature(model, step):
    for i, blk in enumerate(model.transformer.h):
        if not hasattr(blk.attn, "log_c"):
            continue
        c_vals = torch.exp(
            blk.attn.log_c.detach()
            ).cpu().flatten()  # shape (n_heads,)
        for j, c in enumerate(c_vals):
            writer.add_scalar(f"Curvature/layer_{i}/head_{j}", c.item(), step)


if master_process:
    model_size = raw_model.model_size()
    print("\n=== Model ===")
    print(f"Model Size:    {model_size}\n")
    print("Parameter groups:")
    print(f"curv:{n_params(curv_params):,} | "
          f"mat:{n_params(matrix_params):,} | "
          f"non_mat:{n_params(non_matrix_params):,} | "
          f"wte:{n_params(wte_params):,} | "
          f"head:{n_params(head_params):,}\n")
    print(f"Data Path:            {config.data_path}")
    print(f"Sequence Length:      {config.sequence_length}")
    print(f"Total Tokens:      {config.num_iterations * tokens_per_iter:,}")
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

if master_process:
    def create_run_id(config, dataset_name, timestamp):
        """Create a run identifier."""
        dataset_aliases = {
            'shakespeare_char': 'sh',
            'tinystories_char': 'tsc',
            'tinystories': 'ts',
            'taoteching': 'tao',
            'cn_wiki': 'cn',
            'fineweb': 'fw',
            'finewebedu': 'fwe'
        }
        mode_aliases = {
            'euc': 'e',
            'hyp': 'h'
        }
        date = timestamp.strftime('%m.%d')
        seconds_since_midnight = (
            timestamp - timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
                )
                ).seconds
        # get architecture configuration
        head = mode_aliases[config.head_mode]
        attn = mode_aliases[config.attn_mode]
        arch = f"{head}{attn}"
        # build the hyperbolic parameters string
        hyp_params = ""
        if 'eh' in arch:
            if config.k_lr:
                hyp_params += f"_lr{config.k_lr:.0f}"
            elif config.k_lr == 0:
                hyp_params += f"_c{config.curvature:.0f}"
        run_id = (
            f"{seconds_since_midnight:05d}_"
            f"{dataset_aliases[dataset_name]}_"
            f"{arch}{hyp_params}_"
            f"{model_size}_"
            f"s{config.seed}"
        )
        return date, run_id

    # create the run ID
    now = datetime.datetime.now()
    date, run_id = create_run_id(config, dataset_name, now)
    # create log directory and file
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


# initialize CUDA events for timing
total_start = torch.cuda.Event(enable_timing=True)
interval_start = torch.cuda.Event(enable_timing=True)
interval_end = torch.cuda.Event(enable_timing=True)
step_estimates = []
total_start.record()
interval_start.record()

train_loss_accum = 0.0
train_log_count = 0

val_loss_accum = 0.0
val_log_count = 0

best_val_loss = float('inf')

# begin training
train_loader.reset()
for step in range(config.num_iterations + 1):
    last_step = (step == config.num_iterations)
    if (last_step or (config.val_loss_every > 0
                      and step % config.val_loss_every == 0)):
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:  # of course, we'd like to use no_grad() here too,
                # but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_loss_accum += val_loss
        val_log_count += 1
        # log val loss to console

# bit confusing: we want to make sure to eval on 0th iteration
# but also after the very last iteration. so we loop for step <= num_iterations
# instead of just < num_iterations (one extra due to <=), only to do
# the validation/sampling one last time,
# and then we break right here as we're done.
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
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()  # just sync on the last step

    for name, p in model.named_parameters():
        if p.grad is None:
            # print(f"WARNING: Parameter {name} has no gradient. Skipping.")
            continue
        p.grad /= train_accumulation_steps
    # gradient norm monitoring
    if master_process and step % config.train_loss_every == 0:
        gn_curv = grad_norm(curv_params)
        gn_matrix = grad_norm(matrix_params)
        gn_nonmat = grad_norm(non_matrix_params)
        gn_wte = grad_norm(wte_params)
        gn_head = grad_norm(head_params)

        writer.add_scalar('grad_norm/curv',   gn_curv,   step)
        writer.add_scalar('grad_norm/matrix', gn_matrix, step)
        writer.add_scalar('grad_norm/non_mat', gn_nonmat, step)
        writer.add_scalar('grad_norm/wte',    gn_wte,    step)
        writer.add_scalar('grad_norm/head',   gn_head,   step)
        if step % (config.print_multiplier * config.train_loss_every) == 0:
            grads_string = (
                f"Grad norms: curv={gn_curv:.3g} | "
                f"matrix={gn_matrix:.3g} | "
                f"non_mat={gn_nonmat:.3g} | "
                f"wte={gn_wte:.3g} | "
                f"head={gn_head:.3g} | "
            )

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    with torch.no_grad():
        for log_c in curv_params:
            log_c.clamp_(min=math.log(1e-5), max=math.log(1e3))

    model.zero_grad(set_to_none=True)
    train_loss_accum += train_loss.item()
    train_log_count += 1
    # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    # all-reducing the training loss would be
    # more correct in terms of logging, but slower
    if master_process and step % config.train_loss_every == 0:
        # within the main training loop, after
        # logging validation loss or training loss
        interval_end.record()
        torch.cuda.synchronize()

        # Calculate elapsed time in milliseconds
        interval_time_ms = interval_start.elapsed_time(interval_end)
        step_estimates.append(interval_time_ms / config.train_loss_every)
        if len(step_estimates) >= 10:
            avg_time_per_step = sum(step_estimates[-10:]) / 10.
        elif len(step_estimates):
            avg_time_per_step = sum(step_estimates) / len(step_estimates)
        else:
            avg_time_per_step = np.nan

        estimated_total_time = avg_time_per_step * \
            (config.num_iterations - step) / 1e3
        # compute the averages
        avg_train_loss = train_loss_accum / max(1, train_log_count)
        avg_val_loss = val_loss_accum / max(1, val_log_count)

        # log
        tokens_seen = step * tokens_per_iter
        writer.add_scalar('Loss/Train',      avg_train_loss, tokens_seen)
        writer.add_scalar('Loss/Validation', avg_val_loss,   tokens_seen)
        print(
            f"step {step} ({interval_time_ms:.0f}ms):\t"
            f"{tokens_seen/1e6:.1f}M tokens seen,\t"
            f"train loss = {avg_train_loss:.4f},\t"
            f"val loss = {val_loss:.4f},\t"
            f"ETA = {estimated_total_time:.0f}s"
            )
        if config.k_lr and step % (config.print_multiplier *
                                   config.train_loss_every) == 0:
            print_curvature_stats(raw_model.transformer.h, grads_string)
        if config.k_lr and step % config.log_curv_every == 0:
            log_curvature(raw_model, step)
        # reset accumulators
        train_loss_accum = 0.0
        train_log_count = 0
        val_loss_accum = 0.0
        val_log_count = 0

        if config.save_every and \
            (step % config.save_every == 0 or last_step) and \
                avg_val_loss < best_val_loss:
            ckpt = dict(step=step,
                        model=raw_model.state_dict(),
                        optimizers=[opt.state_dict() for opt in optimizers],
                        best_val=avg_val_loss)
            path = f"ckpts/{run_id}_{step:05d}.pt"
            torch.save(ckpt, path)
            best_val_loss = avg_val_loss

        if config.gen_every and master_process and \
           (step % config.gen_every == 0) and (config.gen_first + step):
            context = encode_text(tokenizer, config.gen_prompt, device)
            generated_tokens = raw_model.generate_text(
                context,
                max_length=config.gen_length,
                temperature=1.0,
                top_k=50
                )
            generated_text = decode_tokens(tokenizer, generated_tokens[0])
            writer.add_text(
                f"Generated_Text/Step_{step}", generated_text, step)
            print(
                f"\nGenerated Text: \n{generated_text}\n")
        interval_start.record()
if master_process:
    total_end_event = torch.cuda.Event(enable_timing=True)
    total_end_event.record()
    torch.cuda.synchronize()

    total_time_s = total_start.elapsed_time(total_end_event) / 1e3
    time_msg = f"Total training time: {total_time_s:.2f}s"
    print(time_msg)
    writer.add_text("Time", time_msg, step)
    mem_msg = (
        f"Peak memory consumption: "
        f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    print(mem_msg)
    writer.add_text("GPU", mem_msg, step)

if master_process:
    writer.close()

dist.destroy_process_group()
