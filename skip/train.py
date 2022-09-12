
import parser

import numpy as np
import torch

import dataset
import longrange
import transformer


def train_transformer(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = dataset.get_random_batches(ds, n_batches, batch_size, seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch in pbar:
        batch = batch.to(device)

        loss = transformer.loss_fn(net, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss=loss.item())
        
def train_perceiver(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = dataset.get_random_batches(ds, n_batches, batch_size, seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch in pbar:
        batch = batch.to(device)

        loss = transformer.loss_fn(net, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss=loss.item())

def train_longrange(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = dataset.get_random_skip_batches(ds, n_batches, batch_size, seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch1, batch2 in pbar:
        batch1, batch2 = batch1.to(device), batch2.to(device)

        loss = longrange.loss_fn_longrange(net, batch1, batch2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss=loss.item())
