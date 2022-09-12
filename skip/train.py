
import argparse
import parser

import numpy as np
import torch
import wandb
from tqdm import tqdm

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


def main():
    args = parser.parse_args()
    print(args)
    print(args.n_batches)

    ds_train, ds_test = dataset.create_dataset(tqdm=tqdm)

    if args.model == 'transformer':
        config = transformer.get_config('gpt-nano')
        net = transformer.GPT(config)
        train_fn = train_transformer
    elif args.model == 'perceiver':
        config = transformer.get_config('gpt-nano', n_latent_tokens=30)
        net = transformer.GPT(config)
        train_fn = train_transformer
    elif args.model == 'longrange':
        config = longrange.get_config('gpt-nano')
        net = longrange.LongRangeGPT(config)
        train_fn = train_longrange

    train_fn(ds_train, net, n_batches=args.n_batches, batch_size=args.batch_size,
             seq_len=args.seq_len, device=args.device, tqdm=tqdm)




parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--model', type=str, default='longrange')
parser.add_argument('--n_batches', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--lr', type=float, default=1e-2)

if __name__ == '__main__':
    main()
