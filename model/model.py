import math
import torch
from torch import nn
import torch.nn.functional as F

from lorentz import LorentzManifold
from model.lmath import project, distance

# --------------------------------
# Hyperbolic Attention
# --------------------------------  

# class HyperbolicEmbedding(nn.Module):
#     def __init__(self, vocab_size, dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.spatial_embed = nn.Embedding(vocab_size, dim)
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.normal_(self.spatial_embed.weight, mean=0.0, std=0.02)
        
#     def forward(self, idx):
#         x_spatial = self.spatial_embed(idx)
        
#         # x0 computation
#         spatial_sq = torch.sum(x_spatial**2, dim=-1, keepdim=True)
#         spatial_sq = torch.clamp(spatial_sq, min=0)  # Prevent negative due to numerical errors
#         x0 = torch.sqrt(1 + spatial_sq + self.eps)
        
#         return torch.cat([x0, x_spatial], dim=-1)

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    half_d = x.shape[3]//2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


def custom_attention(query, key, value, curvature=None, 
                                 mode='euc', dropout_p=0.0,
                                 is_causal=False, scale=None,
                                 eps=1e-6, p=2) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if mode == 'euc':
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
    elif mode == 'hyp':
        assert curvature is not None
        lq = project(query, k=curvature, dim=-1).unsqueeze(-2)
        lk = project(key, k=curvature, dim=-1).unsqueeze(-3)
        dis = distance(lq, lk, k=curvature, dim=-1)
        attn_weight = 1 / (eps + dis**p)

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

# - custom_attention: computes causal attention (optionally in 'hyp' mode).

class JointSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.attn_mode
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_heads
        assert self.n_embd % self.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Combined QKV projection (more efficient than separate projections)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        
        if self.mode == 'hyp':
            if not getattr(config, 'k_lr', False):
                self.register_buffer('k', torch.tensor(float(config.curvature)))
            else:
                init_k = torch.exp(torch.randn(1, 1, self.n_heads, 1)) * config.curvature
                self.k = nn.Parameter(init_k)

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)  
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape and transpose: (B, T, n_embd) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Get rotary parameters and apply RMS normalization
        cos, sin = self.rotary(q)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        if self.mode == 'hyp':
            y = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                 is_causal=True, mode='hyp', curvature=self.k)
        else:
            y = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                 is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# original attention implementation
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_heads
        assert self.n_embd % self.n_heads == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

        
class LorentzMLR(nn.Module):
    def __init__(self, num_features, num_classes, curvature=1.0, sparse=False):
        super().__init__()
        # 1) Create a LorentzManifold with a chosen curvature K
        self.manifold = LorentzManifold(K=curvature)

        # 2) Allocate Lorentz parameters:
        #    For multi-class logistic regression, we want 'num_classes' many
        #    hyperbolic "prototypes" (class embeddings) each in dimension num_features.
        #    We actually need dimension = num_features + 1 in the Lorentz model,
        #    but .allocate_lt accounts for that internally.
        self.lt = self.manifold.allocate_lt(num_classes, num_features, sparse=sparse)
        
        # Initialize them properly on the hyperboloid
        self.manifold.init_weights(self.lt)
    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, num_features)
           or (batch_size, num_features) depending on usage
        Return: logits of shape (batch_size, seq_len, num_classes)
                or (batch_size, num_classes)
        
        We'll do the simplest approach: 
        - We interpret each row in `x` as a point in hyperbolic space (by 
          padding or some transform).
        - We measure some "distance" to each class prototype. 
        - Then negative distance can act as the unnormalized logit.
        """
        # 1) Flatten or reshape as needed so we treat each row as an embedding
        B, T, D = x.shape  # if x has 3D shape, e.g. (batch, seq_len, emb_dim)
        # Or if it’s (batch, D) then T=1 is implied.
        x = x.view(B * T, D)

        # 2) On the forward pass, we might “lift” x from R^D to the hyperboloid 
        #    coordinates (D+1) by something like:
        #    new_x = [ sqrt(1 + ||x||^2), x_1, ..., x_D ]
        #    In practice, you can create a small helper function that does the 
        #    same as `Manifold.normalize()` does for a batch of points. 
        #    The code below is a quick demonstration:
        x_expanded = torch.zeros(x.size(0), D + 1, device=x.device)
        # time-like coordinate
        x_expanded[:, 0] = torch.sqrt(1 + (x * x).sum(dim=1))  
        # space-like part
        x_expanded[:, 1:] = x

        # 3) Get the class embeddings from the manifold
        #    shape = (num_classes, D+1)
        c = self.lt.weight  # The prototypes, each on the hyperboloid
        c = self.manifold.normalize(c)

        # 4) Compute Lorentz distance between x_expanded and each class prototype
        #    We'll produce shape (batch*T, num_classes)
        #    Then we can interpret negative distance as logits
        #    (because in hyperbolic space, a “closer” class means a bigger logit).
        #    This might be done by the manifold’s `distance` method in a batched way:
        dist = self.manifold.distance(
            x_expanded.unsqueeze(1),  # shape (batch*T, 1, D+1)
            c.unsqueeze(0)           # shape (1, num_classes, D+1)
        )
        # dist now has shape (batch*T, num_classes).

        # 5) Convert distances to (negative) logits
        logits = -dist  # shape (batch*T, num_classes)

        # Reshape back to (batch_size, seq_len, num_classes) if needed
        logits = logits.view(B, T, -1)
        return logits

    def optim_params(self):
        """
        Return the parameter + Riemannian methods for this layer 
        so it can be used with RiemannianSGD or similar.
        """
        return [
            {
                "params": self.lt.parameters(),
                "rgrad": self.manifold.rgrad,
                "expm": self.manifold.expm,
                "logm": self.manifold.logm,
                "ptransp": self.manifold.ptransp,
            }
        ]


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
        ))

        if config.head_mode == 'euc':
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # stdv = 1. / math.sqrt(config.n_embd)
            # nn.init.uniform_(self.lm_head.weight.data, -stdv, stdv)
            self.lm_head.weight.data.zero_()
        
        elif config.head_mode == 'hyp':
            self.lm_head = LorentzMLR(
                num_features=config.n_embd,
                num_classes=config.vocab_size,
                curvature=config.curvature
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

    def model_size(self):
        """Calculate the model size in millions or thousands, based on parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return f"{total_params / 1e3:.2f}K"
        
## OLD
        
# class CausalSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.n_heads = config.n_heads
#         self.n_embd = config.n_embd
#         self.head_dim = self.n_embd // self.n_heads
#         assert self.n_embd % self.n_heads == 0
#         self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
#         self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
#         self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
#         # output projection
#         self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
#         self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
#         self.rotary = Rotary(self.head_dim)

#     def forward(self, x):
#         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
#         q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
#         k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
#         v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
#         cos, sin = self.rotary(q)
#         q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
#         q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
#         y = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
#         y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
#         y = self.c_proj(y)
#         return y
    
# class HyperbolicSelfAttention(nn.Module):

#     def __init__(self, config):
#         super().__init__()
        
#         self.n_heads = config.n_heads
#         self.n_embd = config.n_embd
#         self.head_dim = self.n_embd // self.n_heads
#         assert self.n_embd % self.n_heads == 0
        
#         # key, query, value projections for all heads, but in a batch
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
#         self.rotary = Rotary(self.head_dim)
        
#         if not config.k_lr:
#             self.register_buffer('k', torch.tensor(float(config.curvature)))
#         else:
#             x = torch.randn(1, 1, config.n_heads, 1, device=self.c_attn.weight.device)
#             init_k = torch.exp(x) * config.curvature
#             self.k = nn.Parameter(init_k)

#     def forward(self, x):
#         B, T, C = x.size()

#         q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
#         q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
#         v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

#         cos, sin = self.rotary(q) 
#         q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
#         q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

#         y = custom_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
#                              is_causal=True, mode='hyp', curvature=self.k)
            
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.c_proj(y)
#         return y
    
## ELDERLY

# class HyperbolicSelfAttention(nn.Module):

#     def __init__(self, config):
#         super().__init__()
        
#         self.n_heads = config.n_heads
#         self.n_embd = config.n_embd
#         self.head_dim = self.n_embd // self.n_heads
#         assert self.n_embd % self.n_heads == 0
        
#         # key, query, value projections for all heads, but in a batch
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
#         if not config.k_lr:
#             self.register_buffer('c', torch.tensor(float(config.curvature)))
#         else:
#             x = torch.randn(1, config.n_heads, 1, 1, device=self.c_attn.weight.device)
#             init_k = torch.exp(x) * config.curvature
#             self.k = nn.Parameter(init_k)
        
#         self.register_buffer('p', torch.tensor(2.0))
#         self.register_buffer('eps', torch.tensor(1e-3))
#         self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length)).view(1, 1, config.sequence_length, config.sequence_length))

#     def forward(self, x):
#         B, T, C = x.size()

#         q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
#         q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
#         v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

#         lq = project(q, k=self.k, dim=-1).unsqueeze(-2)
#         lk = project(k, k=self.k, dim=-1).unsqueeze(-3)

#         dis = distance(lq, lk, k=self.k, dim=-1)

#         wei = 1 / (self.eps + dis**self.p)
#         wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, 0.) 
#         wei = wei / wei.sum(dim=-1, keepdim=True)
#         y = wei @ v
            
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.c_proj(y)
#         return y