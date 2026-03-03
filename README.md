# Hyperbolic nanoGPT

Code for *Curvature Stratification in Attention: Probing Intrinsic Geometry* (arXiv link TBD). The repo preserves the full history of experiments and tuning; bringing it to "reproduce experiments out of the box" is work in progress.

## Modifications

**Hyperbolic attention** — QK similarity is based on hyperbolic (Lorentz) distance instead of dot product, with a learnable curvature param per attention head. Set `attn_mode='hyp'` and `k_lr > 0` to enable.

**Lorentz LM Head** — WIP; not required for the paper. Experiments use `head_mode='euc'`.

## Installation and Running

**Main env** (full training):

```bash
conda env create -f env.yaml
conda activate hypgpt
```

**Light env** (debug/tests, CPU):

```bash
conda create -n hypgpt-test python=3.10 -y
conda activate hypgpt-test
pip install torch transformers
```

**Debug scripts** (no GPU):

- `python debug/debug.py` — forward pass, backward, generation (Shakespeare char)
- `python debug/debug_inference.py` — load FineWebEdu checkpoint and generate (uses `data/gpt2_tokenizer`)

**Experiment scripts** (GPU, DDP):

- `run/shakespeare.sh`, `run/shakespeare_head.sh`
- `run/tinystories.sh`, `run/tinystories_char.sh`
- `run/fineweb.sh`, `run/fineweb_euc.sh`
- `run/test.sh` — short run for sanity check

Run from repo root with `torchrun`, e.g.:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt2.py --data_path data/shakespeare_char --attn_mode hyp --head_mode euc --k_lr 1.0 ...
```

## Key Parameters


| Parameter                                            | Description                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `attn_mode`                                          | `'hyp'` (hyperbolic) or `'euc'` (baseline)                   |
| `head_mode`                                          | `'hyp'` or `'euc'` for LM head                               |
| `k_lr`                                               | Learning rate for curvature (0 = fixed)                      |
| `curvature`                                          | Initial curvature                                            |
| `data_path`                                          | Dataset dir, e.g. `data/shakespeare_char`, `data/finewebedu` |
| `n_layers`, `n_heads`, `head_dim`, `sequence_length` | Architecture                                                 |


## Acknowledgements

- [modded-nanogpt](https://github.com/kellerjordan/nanoGPT) and [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) for the base implementation.

