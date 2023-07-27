import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, bias=bias, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def attn_forward(self, x, mask=None):
        # x, _ = self.attn(x, x, x)
        x, self.attn_weights = self.attn(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False)
        return x

    def forward(self, x, mask=None):
        x = x + self.attn_forward(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * n_layer))

        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tok):
        device = tok.device
        b, t = tok.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.wte(tok)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x, mask=self.mask[:t, :t])
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
