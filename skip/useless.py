import math
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from sinusoidal import calc_fourier_position_embeddings

default_config = dict(
    n_vocab=128,
    max_context_length=None,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
    n_layer=3,
    n_head=3,
    n_embd=48,
    n_latent_tokens=None,
)
configs = {
    "gpt1": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
    "gpt-ak": dict(n_layer=6, n_head=6, n_embd=384),
}


def get_config(premade=None, **kwargs):
    config = default_config.copy()
    if premade is not None:
        config.update(configs[premade])
    config.update(kwargs)
    return config

# Code from https://ai.stackexchange.com/questions/35548/when-exactly-does-the-split-into-different-heads-in-multi-head-attention-occur
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        if d_k is None:
            d_k = self.d_model//self.n_heads
            d_v = self.d_model//self.n_heads
        
        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_heads * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        
    def get_mask(self, attn, alibi=None, mask='full'):
        _, bs, nx, ny = attn.shape # output, input
        if mask == 'full':
            mask = torch.ones(bs, nx, ny, dtype=bool, device=attn.device)
        if mask == 'causal':
            mask = ~torch.tril(torch.ones(nx, ny, dtype=bool, device=attn.device), diagonal=ny-nx)
        if mask == 'doublecausal':
            # cross attention but don't attend to stuff ahead in the other seq
            ny = nx
            mask = ~torch.tril(torch.ones(nx, ny, dtype=bool, device=attn.device), diagonal=ny-nx)
            mask = torch.cat([mask, mask], dim=-1)

    def forward(self, query, key, value, mask='full'):
        q = rearrange(self.w_qs(query), 'b l (head q) -> head b l q', head=self.n_heads)
        k = rearrange(self.w_ks(key), 'b t (head k) -> head b t k', head=self.n_heads)
        v = rearrange(self.w_vs(value), 'b t (head v) -> head b t v', head=self.n_heads)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        
        
        if mask is not None:
            if mask.ndim==2:
                mask = mask[None]
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        
        return output, attn

# Code from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        # self.attn = nn.MultiheadAttention(
        #     config["n_embd"],
        #     config["n_head"],
        #     dropout=config["attn_pdrop"],
        #     batch_first=True,
        # )
        self.attn = MultiHeadAttention(
            config["n_embd"],
            config["n_head"],
            dropout=config["attn_pdrop"],
            # batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["resid_pdrop"]),
        )
        self.attn_weights = None

    def forward(self, x, y=None, mask='full'):
        if y is None:
            y = x
        # print(mask)
        if mask is None or mask == 'full':
            mask = None
        if mask == 'causal':
            nx, ny = x.shape[-2], y.shape[-2]
            mask = ~torch.tril(torch.ones(nx, ny, device=x.device), diagonal=ny-nx).to(bool)
        if mask == 'doublecausal':
            # cross attention but don't attend to stuff ahead in the other seq
            nx, ny = x.shape[-2], y.shape[-2]
            ny = nx
            mask = ~torch.tril(torch.ones(nx, ny, device=x.device), diagonal=ny-nx).to(bool)
            mask = torch.cat([mask, mask], dim=-1)
            # plt.imshow(mask.detach().cpu().numpy())
            # plt.show()
            # print('akakak')
            # print(mask.shape)

        lnx, lny = self.ln_1(x), self.ln_1(y)
        
        # attn_output, attn_weights = self.attn(query=lnx, key=lny, value=lny, need_weights=True, attn_mask=mask, average_attn_weights=False)
        attn_output, attn_weights = self.attn(query=lnx, key=lny, value=lny, mask=mask)
        self.attn_weights = attn_weights.detach().clone()
        # print(attn_weights.shape, self.wtf.shape)
        # x = x + self.attn(query=lnx, key=lny, value=lny, need_weights=False, attn_mask=mask)[0]
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config["n_vocab"], config["n_embd"])
        if config["max_context_length"] is None:
            self.wpe = partial(calc_fourier_position_embeddings, config=self.config)
        else:
            self.wpe = nn.Embedding(config["block_size"], config["n_embd"])
        self.drop = nn.Dropout(config["embd_pdrop"])
        self.blocks = nn.ModuleList(
            [Block(config) for idx_layer in range(config["n_layer"])]
        )
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lin_head = nn.Linear(config["n_embd"], config["n_vocab"], bias=False)

    def forward(self, ids):
        device = ids.device
        bs, context_length = ids.size()  # batch size, context length
        mcl = self.config["max_context_length"]
        assert mcl is None or context_length <= mcl

        pos = torch.arange(0, context_length, dtype=torch.long, device=device)[None]
        # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(ids)  # (b, cl, n_embd)
        pos_emb = self.wpe(pos)  # (1, cl, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # print(x.shape)
        if self.config['n_latent_tokens'] is None:
            x = self.blocks[0](x, mask='causal')
        else:
            nlt = self.config['n_latent_tokens']
            x = self.blocks[0](x=x[:, -nlt:, :], y=x, mask='causal')
        # print(x.shape)
        for block in self.blocks[1:]:
            x = block(x, mask='causal')
            # print(x.shape)

        x = self.ln_f(x)
        logits = self.lin_head(x)
        # print(logits.shape)
        return logits

fn_cross_entropy = nn.CrossEntropyLoss()


def loss_fn(net, ids):
    inputs, targets = ids[..., :-1], ids[..., 1:]  # ..., context_length-1
    logits = net(inputs)  # -1, context_length-1, n_vocab
    targets = targets[:, -logits.size(1) :] # -1, however many outputs we have
    loss = fn_cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss

@torch.no_grad()
def generate(
    model,
    ids,
    n_tokens,
    max_context_length=1000,
    temperature=1.0,
    do_sample=False,
    top_k=None,
):
    """
    Take a conditioning sequence of indices ids (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(n_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        if max_context_length is not None:
            ids_cond = ids[:, -max_context_length:]
        # forward the model to get the logits for the index in the sequence
        logits = model(ids_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            ids_next = torch.multinomial(probs, num_samples=1)
        else:
            _, ids_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        ids = torch.cat((ids, ids_next), dim=1)

    return ids
