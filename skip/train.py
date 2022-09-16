
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
        wandb.log(dict(loss=loss.item()))

def train_longrange(ds, net, n_batches=10, batch_size=32, seq_len=100, lr=1e-2, device='cpu', tqdm=None):
    net = net.to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    pbar = dataset.get_random_skip_batches(ds, n_batches, batch_size, seq_len, min_dist=4*seq_len, max_dist=100*seq_len)
    pbar = pbar if tqdm is None else tqdm(pbar, total=n_batches)
    for batch1, batch2 in pbar:
        batch1, batch2 = batch1.to(device), batch2.to(device)

        loss = longrange.loss_fn_longrange(net, batch1, batch2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss=loss.item())
        wandb.log(dict(loss=loss.item()))


def main():
    args = parser.parse_args()
    print(args)
    wandb.init(config=args)

    ds_train, ds_test = dataset.load_datasets(tqdm=tqdm)

    if args.model == 'transformer':
        config = transformer.get_config('gpt-ak')
        net = transformer.GPT(config)
        train_fn = train_transformer
    elif args.model == 'perceiver':
        config = transformer.get_config('gpt-ak', n_latent_tokens=args.n_latent_tokens)
        net = transformer.GPT(config)
        train_fn = train_transformer
    elif args.model == 'longrange':
        config = longrange.get_config('gpt-ak')
        net = longrange.LongRangeGPT(config)
        train_fn = train_longrange
        
    wandb.config.update(config)
    
    print('Model config: ')
    print(config)
    print(f'Model # of parameters: {util.count_params(net)}')

    train_fn(ds_train, net, n_batches=args.n_batches, batch_size=args.batch_size,
             seq_len=args.seq_len, device=args.device, tqdm=tqdm)


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--model', type=str, default='longrange')
parser.add_argument('--n_batches', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--n_latent_tokens', type=int, default=None)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--lr', type=float, default=1e-2)

if __name__ == '__main__':
    main()

"""
python train.py --device cuda:0 --model transformer --seq_len 64 --seed 0

python train.py --device cuda:1 --model perceiver --seq_len 128 --seed 0
python train.py --device cuda:2 --model perceiver --seq_len 256 --seed 0
python train.py --device cuda:0 --model perceiver --seq_len 512 --seed 0

python train.py --device cuda:3 --model longrange --seq_len 64 --batch_size 64 --seed 0
"""
