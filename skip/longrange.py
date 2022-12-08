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

configs = {
    "transformer": dict(n_layer=6, n_head=6, n_embd=384),
    "perceiver": dict(n_layer=6, n_head=6, n_embd=384, n_latent_tokens=64),
    "longrange1": dict(n_layer=6, n_head=6, n_embd=384, use_memory=True, share_cr=False),
    "longrange2": dict(n_layer=6, n_head=6, n_embd=384, use_memory=True, share_cr=True),
    "longrange3": dict(n_layer=6, n_head=6, n_embd=384, use_memory=True, share_cr=False, memory_cross_attn_only=False),
}

def get_config(premade=None, **kwargs):
    config = default_config.copy()
    if premade is not None:
        config.update(configs[premade])
    if 'use_memory' not in config:
        config['use_memory'] = False
    if 'share_cr' not in config:
        config['share_cr'] = False
    if 'memory_cross_attn_only' not in config:
        config['memory_cross_attn_only'] = True
    config.update(kwargs)
    return config

def calc_attn_mask(d_out, d_in, mask='causal', alibi=False, dtype=None, device='cpu'):
    """
    Return 1, d_out, d_in shape with -inf where attn cannot happen.
    (the 1 corresponds to the different heads)
    
    During cross attn it should be used as cross attn with A and cat([B, A])
    """
    if mask == 'full':
        mask = torch.ones(d_out, d_in, dtype=bool, device=device)
    if mask == 'causal':
        mask = torch.tril(torch.ones(d_out, d_in, dtype=bool, device=device), diagonal=d_in-d_out)
    if mask == 'doublecausal':
        # cross attention but don't attend to stuff ahead in the other seq
        d_in = d_out
        mask = torch.tril(torch.ones(d_out, d_in, dtype=bool, device=device), diagonal=d_in-d_out)
        mask = torch.cat([mask, mask], dim=-1)
    a = torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, value=-torch.inf)
    return a[None]
def calc_attn_mask_with_attn(attn, mask='causal', alibi=False):
    d_out, d_in = attn.shape[-2:]
    return calc_attn_mask(d_out, d_in, mask=mask, alibi=alibi, dtype=attn.dtype, device=attn.device)

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
        

    def forward(self, query, key, value, mask='causal', alibi=False):
        q = rearrange(self.w_qs(query), 'b l (head q) -> b head l q', head=self.n_heads)
        k = rearrange(self.w_ks(key), 'b t (head k) -> b head t k', head=self.n_heads)
        v = rearrange(self.w_vs(value), 'b t (head v) -> b head t v', head=self.n_heads)
        attn = torch.einsum('bhlk,bhtk->bhlt', [q, k]) / np.sqrt(q.shape[-1])
        
        mask = calc_attn_mask_with_attn(attn, mask=mask, alibi=alibi)
        attn = attn+mask[None]
        
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        output = rearrange(output, 'b head l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        
        # for debugging
        self.attn_mask = mask.detach()
        self.attn_mat = attn.detach()
        
        return output, attn

# Code from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.mhattn = MultiHeadAttention(
            config["n_embd"],
            config["n_head"],
            dropout=config["attn_pdrop"],
        )
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["resid_pdrop"]),
        )

    def forward(self, x, y=None, mask='causal', alibi=False):
        """
        x is the output sequence (queries generated from this)
        y is the input sequence (keys+values generated from this)
        """
        if y is None:
            y = x
        lnx, lny = self.ln_1(x), self.ln_1(y)
        attn_output, attn_mat = self.mhattn(query=lnx, key=lny, value=lny, mask=mask, alibi=alibi)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        
        # for debugging
        self.attn_mat = attn_mat.detach()
        
        return x


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
            self.blocks_memory_retriever = nn.ModuleList(
                [Block(config) for idx_layer in range(nb_3*2)]
            )

            # share_memory_creator_retriever
        self.share_cr = config['share_cr']
        self.memory_cross_attn_only = config['memory_cross_attn_only']
        
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lin_head = nn.Linear(config["n_embd"], config["n_vocab"], bias=False)
        
        self.fn_cross_entropy = nn.CrossEntropyLoss()

    # def forward(self, ids, long_range_input=None, calc_long_range_output=False):
    def forward(self, ids, memory_in=None, calc_memory_out=False, use_my_lrr_kv=False):
        self.ids = ids
        # use_longterm_stack = use_my_lrr_kv or lrr_memory is not None
        
        device = ids.device
        bs, context_length = ids.size() # batch size, context length
        mcl = self.config["max_context_length"]
        assert mcl is None or context_length <= mcl

        pos = torch.arange(0, context_length, dtype=torch.long, device=device)[None]
        # shape (1, t)

        tok_emb = self.wte(ids)  # (b, cl, n_embd)
        pos_emb = self.wpe(pos)  # (1, cl, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        nlt = x.shape[-2] if self.config['n_latent_tokens'] is None else self.config['n_latent_tokens']

        nb_3 = len(self.blocks_main) // 3
        blocks1, blocks2, blocks3 = [self.blocks_main[i*nb_3:(i+1)*nb_3] for i in range(3)]
        if self.config['use_memory']:
            nbm_2 = len(self.blocks_memory_retriever) // 2
            blocks_mr1, blocks_mr2 = [self.blocks_memory_retriever[i*nbm_2:(i+1)*nbm_2] for i in range(2)]
            

        # print('pre first ', x.shape)
        x = blocks1[0](x=x[:, -nlt:, :], y=x, mask='causal') # Perceiver-AR Block
        # print('post first ', x.shape)
        for block in blocks1[1:]: # Main transformer first third
            x = block(x, mask='causal')
        # print('main - First third')
        # print(x.shape)

        memory_retreival = x
        if (self.share_cr and calc_memory_out) or (memory_in is not None):
            for block in blocks_mr1: # Memory retreival first half
                memory_retreival = block(memory_retreival, mask='causal')
            # print('retreival - First half')
            # print(memory_retreival.shape)
        if calc_memory_out: # Create memory output
            memory_created = memory_retreival if self.share_cr else x # share processing step if needed
            for block in self.blocks_memory_creator: # memory output blocks
                memory_created = block(memory_created, mask='full')
            # print(f'creation - using {self.share_cr=}')
            # print(memory_created.shape)

        if memory_in is not None: # Use memory input
            # Merge in memory input where local context ONLY gives queries
            # print()
            # print(memory_retreival.shape, memory_in.shape)
            if self.memory_cross_attn_only:
                memory_retreival = blocks_mr2[0](x=memory_retreival, y=memory_in, mask='full')
            else:
                memory_retreival = blocks_mr2[0](x=memory_retreival, y=torch.cat([memory_retreival, memory_in], dim=-2), mask='causal')
            # print('cross attn memory retrieval and memory_in')
            # print(memory_retreival.shape)
            # print()
            for block in blocks_mr2[1:]: # Memory retreival second half
                memory_retreival = block(memory_retreival, mask='causal')
            # print('retreival - second half')
            # print(memory_retreival.shape)

        for block in blocks2: # Main transformer second third
            x = block(x, mask='causal')
        # print('main - second third')
        # print(x.shape)

        if memory_in is not None: # Merge memory into main
            x = blocks3[0](x=x, y=torch.cat([memory_retreival, x], dim=-2), mask='doublecausal')
            # print('merging of memory!')
            # print(x.shape)
        else:
            x = blocks3[0](x, mask='causal') # Don't merge memory into main
            # print('no merging of memory')
            # print(x.shape)
        for block in blocks3[1:]: # Main transformer third/last third
            x = block(x, mask='causal')
        # print('main - third third')
        # print(x.shape)

        x = self.ln_f(x)
        logits = self.lin_head(x)

        return (logits, memory_created) if calc_memory_out else (logits, None)


def loss_fn_longrange(net, ids1, ids2):
    fn_cross_entropy = net.fn_cross_entropy
    
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
    fn_cross_entropy = net.fn_cross_entropy
    inputs, targets = ids[..., :-1], ids[..., 1:]  # ..., context_length-1
    logits, _ = net.forward(inputs)  # -1, context_length-1, n_vocab
    targets = targets[:, -logits.size(1) :] # -1, however many outputs we have
    loss = fn_cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss

def loss_fn_longrange(net, batch_ids, batch_fbin):
    """
    Computes losses using long-range memory
    - batch_ids is of shape (bs, n_seqs, seq_len)
    - batch_fbin is of shape (bs, n_seqs, seq_len)
    
    Returns 
    - losses of shape bs, n_seqs, seq_len-1
    - fbin2loss is dictionary from fbin to [loss for each seq in n_seqs]
    """
    fn_cross_entropy = net.fn_cross_entropy
    bs, n_seqs, seq_len = batch_ids.shape
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    fbin2loss = {fbin: torch.full((n_seqs, ), torch.nan) for fbin in range(-2, 9)}
    losses = []
    import parser
    
    memory = None
    for idx_seq, (ids_seq, fbin_seq) in enumerate(zip(batch_ids.unbind(dim=-2), batch_fbin.unbind(dim=-2))):
        fbin_seq = fbin_seq[..., 1:]
        inputs, targets = ids_seq[..., :-1], ids_seq[..., 1:] # ..., context_length-1
        logits, memory_out = net.forward(inputs, memory_in=memory, calc_memory_out=net.config['use_memory'])
        memory = memory_out if memory is None else torch.cat([memory, memory_out], dim=-2)
        
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).reshape(bs, -1)
        losses.append(loss)
        
        for fbin in fbin_seq.unique().sort().values.tolist():
            fbin2loss[fbin][idx_seq] = loss.flatten()[fbin_seq.flatten()==fbin].mean().item()
    losses = torch.stack(losses, dim=-2)
    return losses, fbin2loss

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

