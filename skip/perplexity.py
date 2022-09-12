from collections import defaultdict

import torch
from torch import nn


def calc_perplexity_transformer(ds, net, seq_len, stride=1, device='cpu', tqdm=None):
    net.to(device)
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    id2losses = defaultdict(lambda : [])
    for book, ids in ds.items() if tqdm is None else tqdm(ds.items()):
        ids = ids.to(device)
        pbar = range(0, len(ids)-seq_len, stride)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
        for cidx in pbar:
            context_input = ids[cidx:cidx+seq_len]
            context_target = ids[cidx+1:cidx+seq_len+1]
            outputs = net(context_input[None])[0]
            loss = loss_fn(outputs, context_target)
            id2losses[context_target[-1].item()].append(loss[-1].item())
    return id2losses

def calc_perplexity_longrange(ds, net, seq_len, stride=1, device='cpu', tqdm=None):
    net.to(device)
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    id2losses = defaultdict(lambda : [])
    for book, ids in ds.items() if tqdm is None else tqdm(ds.items()):
        ids = ids.to(device)
        pbar = range(0, len(ids)-seq_len, stride)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
        long_range_memory = torch.zeros((1, 0, 48))
        for cidx in pbar:
            context_input = ids[cidx:cidx+seq_len]
            context_target = ids[cidx+1:cidx+seq_len+1]
            outputs, lro = net(context_input[None], long_range_input=long_range_memory, calc_long_range_output=True)[0]
            long_range_memory = torch.cat([long_range_memory, lro], dim=-2)
            loss = loss_fn(outputs, context_target)
            id2losses[context_target[-1].item()].append(loss[-1].item())
    return id2losses
