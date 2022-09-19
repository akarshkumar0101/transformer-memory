import math
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from transformer import (Block, calc_fourier_position_embeddings,
                         fn_cross_entropy)

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

configs = {
    "transformer": dict(n_layer=6, n_head=6, n_embd=384),
    "perceiver": dict(n_layer=6, n_head=6, n_embd=384, n_latent_tokens=128),
    "longrange1": dict(n_layer=6, n_head=6, n_embd=384, use_memory=True, share_cr=False),
    "longrange2": dict(n_layer=6, n_head=6, n_embd=384, use_memory=True, share_cr=True),
}

def get_config(premade=None, **kwargs):
    config = default_config.copy()
    if premade is not None:
        config.update(configs[premade])
    if 'use_memory' not in config:
        config['use_memory'] = False
    if 'share_cr' not in config:
        config['share_cr'] = False
    config.update(kwargs)
    return config


# Code from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class LongRangeGPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, **config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config["n_vocab"], config["n_embd"])
        if config["max_context_length"] is None:
            self.wpe = partial(calc_fourier_position_embeddings, config=self.config)
        else:
            self.wpe = nn.Embedding(config["block_size"], config["n_embd"])

        self.drop = nn.Dropout(config["embd_pdrop"])
        
        assert config['n_layer']%3 == 0
        nb_3 = config['n_layer']//3
        self.blocks_main = nn.ModuleList(
            [Block(config) for idx_layer in range(config["n_layer"])]
        )
        
        if config['use_memory']:
            # outputs a long-range memory representation
            self.blocks_memory_creator = nn.ModuleList(
                [Block(config) for idx_layer in range(nb_3)]
            )
            # inputs a long-range memory representation
            self.blocks_memory_retreiver = nn.ModuleList(
                [Block(config) for idx_layer in range(nb_3)]
            )

            # share_memory_creator_retreiver
        self.share_cr = config['share_cr']
        
        
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lin_head = nn.Linear(config["n_embd"], config["n_vocab"], bias=False)

    # def forward(self, ids, long_range_input=None, calc_long_range_output=False):
    def forward(self, ids, memory_in=None, calc_memory_out=False, use_my_lrr_kv=False):
        # use_longterm_stack = use_my_lrr_kv or lrr_memory is not None
        
        device = ids.device
        bs, context_length = ids.size() # batch size, context length
        mcl = self.config["max_context_length"]
        assert mcl is None or context_length <= mcl

        pos = torch.arange(0, context_length, dtype=torch.long, device=device)[None]
        # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(ids)  # (b, cl, n_embd)
        pos_emb = self.wpe(pos)  # (1, cl, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        nlt = x.shape[-2] if self.config['n_latent_tokens'] is None else self.config['n_latent_tokens']

        nb_3 = len(self.blocks_main) // 3
        blocks1, blocks2, blocks3 = [
            self.blocks_main[i * nb_3 : (i + 1) * nb_3] for i in range(3)
        ]
        if self.config['use_memory']:
            nbm_2 = len(self.blocks_memory_retreiver) // 2
            blocks_mr1, blocks_mr2 = [
                self.blocks_main[i * nbm_2 : (i + 1) * nbm_2] for i in range(2)
            ]

        x = blocks1[0](x=x[:, -nlt:, :], y=x, mask='causal') # Perceiver-AR Block
        for block in blocks1[1:]: # Main transformer first third
            x = block(x, mask='causal')
        # print('main - First third')

        memory_retreival = x
        if (self.share_cr and calc_memory_out) or (memory_in is not None):
            for block in blocks_mr1: # Memory retreival first half
                memory_retreival = block(memory_retreival, mask='causal')
            # print('retreival - First half')
        if calc_memory_out: # Create memory output
            memory_created = memory_retreival if self.share_cr else x # share processing step if needed
            for block in self.blocks_memory_creator: # memory output blocks
                memory_created = block(memory_created, mask='full')
            # print(f'creation - using {share=}')

        if memory_in is not None: # Use memory input
            # Merge in memory input where local context ONLY gives queries
            memory_retreival = blocks_mr2[0](x=memory_retreival, y=memory_in, mask='full')
            for block in blocks_mr2[1:]: # Memory retreival second half
                memory_retreival = block(memory_retreival, mask='causal')
            # print('retreival - second half')

        for block in blocks2: # Main transformer second third
            x = block(x, mask='causal')
        # print('main - second third')

        if memory_in is not None: # Merge memory into main
           x = blocks3[0](x=x, y=torch.cat([memory_retreival, x], dim=-2), mask='causal')
           # print('merging of memory!')
        else:
           x = blocks3[0](x, mask='causal') # Don't merge memory into main
           # print('no merging of memory')
        for block in blocks3[1:]: # Main transformer third/last third
            x = block(x, mask='causal')
        # print('main - third third')

        x = self.ln_f(x)
        logits = self.lin_head(x)

        return (logits, memory_created) if calc_memory_out else (logits, None)


def loss_fn_longrange(net, ids1, ids2):
    inputs1, targets1 = ids1[..., :-1], ids1[..., 1:]  # ..., context_length-1
    inputs2, targets2 = ids2[..., :-1], ids2[..., 1:]  # ..., context_length-1

    logits1, memory1 = net.forward(inputs1, memory_in=None, calc_memory_out=True)
    logits2, memory2 = net.forward(inputs2, memory_in=memory1, calc_memory_out=False)
    # -1, context_length-1, n_vocab
    loss1 = fn_cross_entropy(
        logits1.reshape(-1, logits1.size(-1)), targets1.reshape(-1)
    )
    loss2 = fn_cross_entropy(
        logits2.reshape(-1, logits2.size(-1)), targets2.reshape(-1)
    )
    loss = (loss1 + loss2) / 2.0
    return loss, loss1, loss2

def loss_fn(net, ids):
    inputs, targets = ids[..., :-1], ids[..., 1:]  # ..., context_length-1
    logits, _ = net.forward(inputs)  # -1, context_length-1, n_vocab
    targets = targets[:, -logits.size(1) :] # -1, however many outputs we have
    loss = fn_cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss


@torch.no_grad()
def generate_ak(
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
        ids_cond = ids if max_context_length is None else ids[:, -max_context_length:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(ids_cond)
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

