
import argparse
import parser

import numpy as np
import torch
import wandb
from tqdm import tqdm

import dataset
import longrange
import transformer
import util

def train_transformer(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-3, device='cpu', wandb=None, tqdm=None):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs=1, seq_len=seq_len, min_dist=0, max_dist=1, unbind=True)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for (batch1,), (batch1fchar,) in pbar:
        batch1 = batch1.to(device).long()

        loss = longrange.loss_fn(net, batch1)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(dict(loss=loss.item()))
        if wandb is not None:
            wandb.log(dict(loss=loss.item()))

import akutil

def train_longrange(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-3, device='cpu', wandb=None, tqdm=None):
    akl = akutil.AKLog()
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.MultiStepLR(opt, milestones=[1000], gamma=0.1)


    # pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs=2, seq_len=seq_len, min_dist=0, max_dist=seq_len, unbind=True)
    pbar = dataset.get_seq_batches(ds, n_batches, batch_size, n_seqs=2, seq_len=seq_len, min_dist=0, max_dist=1, unbind=False, device=device)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    # for (batch1, batch2), (batch1fchar, batch2fchar) in pbar:
    for idx_batch, (batch_ids, batch_fbin) in enumerate(pbar):
        # batch1, batch2 = batch1.to(device).long(), batch2.to(device).long()

        losses, _ = longrange.loss_fn_longrange(net, batch_ids, batch_fbin)
        loss1, loss2 = losses[:, 0].mean(), losses[:, 1].mean()
        loss = losses.mean()
        opt.zero_grad()
        loss.backward()
        
        log_network_stats(akl, net)
        
        opt.step()
        
        akl.put(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item())
        if tqdm is not None:
            pbar.set_postfix(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
            # pbar.set_postfix(dict(loss=loss.item()))
        if wandb is not None:
            wandb.log(dict(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item()))
            
        akl.next_ts()
    return akl
            
def log_network_stats(akl, net):
    grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_main]
    akl.put(grad_blocks_main=grad_mag)
    grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_memory_retriever]
    akl.put(grad_blocks_retriever=grad_mag)
    grad_mag = [block.attn.out_proj.weight.grad.norm().item() for block in net.blocks_memory_creator]
    akl.put(grad_blocks_creator=grad_mag)
    if akl.ts%20==0:
        for name, module in net.named_modules():
            if isinstance(module, transformer.Block):
                key = f'attn_mat_{name}'
                akl.put(**{key: module.attn_weights.detach().cpu().numpy()})
    
def print_network_gradient_stats(net):
    print('Main blocks gradient mag:')
    for block in net.blocks_main:
        gradmag = block.attn.out_proj.weight.grad.norm().item()
        print(f'{gradmag: .3e}', end=' ')
    print()
    print('Memory Retrieval blocks gradient mag:')
    for block in net.blocks_memory_retriever:
        gradmag = block.attn.out_proj.weight.grad.norm().item()
        print(f'{gradmag: .3e}', end=' ')
    print()
    print('Memory Creator blocks gradient mag:')
    for block in net.blocks_memory_creator:
        gradmag = block.attn.out_proj.weight.grad.norm().item()
        print(f'{gradmag: .3e}', end=' ')
    print()
    