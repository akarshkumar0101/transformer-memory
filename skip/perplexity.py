from collections import defaultdict

import torch
from torch import nn


def calc_perplexity_transformer(ds, net, seq_len, device='cpu', tqdm=None):
    net.to(device)
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    id2losses = defaultdict(lambda : [])
    for book, ids in ds.items() if tqdm is None else tqdm(ds.items()):
        ids = ids.to(device)
        pbar = range(0, len(ids)-seq_len)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
        for cidx in pbar:
            context_input = ids[cidx:cidx+seq_len]
            context_target = ids[cidx+1:cidx+seq_len+1]
            outputs = net(context_input[None])[0]
            loss = loss_fn(outputs, context_target)
            id2losses[context_target[-1].item()].append(loss[-1].item())
    print(id2losses)
