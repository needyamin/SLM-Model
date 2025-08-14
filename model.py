# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.nh, self.dk).permute(2,0,3,1,4)  # 3, B, nh, T, dk
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, nh, T, dk)
        # scaled dot-product
        att = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.dk)  # (B, nh, T, T)
        if self.mask is None or self.mask.size(2) < T:
            # build causal mask
            device = x.device
            mask = torch.tril(torch.ones((T, T), device=device)).unsqueeze(0).unsqueeze(0)  # 1,1,T,T
            self.mask = mask
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
        w = F.softmax(att, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", w, v)  # B, nh, T, dk
        y = y.permute(0,2,1,3).contiguous().view(B, T, C)
        return self.out(y)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, eps=1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, n_layers=12, d_model=512, n_heads=8, d_ff=2048, seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.seq_len, "Sequence length > seq_len"
        device = idx.device
        tok = self.tok_emb(idx)  # B,T,C
        pos = self.pos_emb[:, :T, :].to(device)
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
